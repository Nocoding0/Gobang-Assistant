from __future__ import annotations

import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .analysis import AnalysisResult, CandidateMove, HeuristicAnalyzer, ProofStatus
from .domain import BoardState, Stone


COORDINATE_RE = re.compile(r"(?<!\d)(\d{1,2}),(\d{1,2})(?!\d)")
MULTIPV_RE = re.compile(r"^MESSAGE\s+\((\d+)\)\s+(-?\d+)")


class EngineProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class RapfiConfig:
    executable: Path
    time_ms: int = 1000
    threads: int = 4
    hash_kib: int = 256 * 1024
    startup_timeout_s: float = 25.0


def parse_rapfi_output(
    output: str, board: BoardState, limit: int = 3
) -> tuple[CandidateMove, ...]:
    """Extract ranked candidate moves from Rapfi Yixin detail messages."""

    found: list[CandidateMove] = []
    seen: set[tuple[int, int]] = set()
    fallback_point: tuple[int, int] | None = None

    for raw_line in output.splitlines():
        line = raw_line.strip()
        coordinate_matches = COORDINATE_RE.findall(line)
        if line and re.fullmatch(r"\d{1,2},\d{1,2}", line):
            fallback_point = tuple(map(int, line.split(",")))

        match = MULTIPV_RE.match(line)
        if not match or not coordinate_matches:
            continue
        rank = int(match.group(1))
        score = int(match.group(2))
        x, y = map(int, coordinate_matches[0])
        if not board.in_bounds(x, y) or board.at(x, y) is not Stone.EMPTY or (x, y) in seen:
            continue
        proof = ProofStatus.FORCED_WIN if abs(score) >= 29_000 else ProofStatus.HEURISTIC
        pv = tuple((int(px), int(py)) for px, py in coordinate_matches)
        found.append(
            CandidateMove(x=x, y=y, rank=rank, score=score, proof=proof, principal_variation=pv)
        )
        seen.add((x, y))

    if fallback_point and fallback_point not in seen:
        x, y = fallback_point
        if board.in_bounds(x, y) and board.at(x, y) is Stone.EMPTY:
            found.append(
                CandidateMove(
                    x=x,
                    y=y,
                    rank=len(found) + 1,
                    score=None,
                    proof=(
                        ProofStatus.WIN_IN_ONE
                        if board.place(x, y).winner() is board.side_to_move()
                        else ProofStatus.HEURISTIC
                    ),
                    principal_variation=((x, y),),
                )
            )

    found.sort(key=lambda move: move.rank)
    return tuple(
        CandidateMove(
            x=move.x,
            y=move.y,
            rank=index,
            score=move.score,
            proof=move.proof,
            principal_variation=move.principal_variation,
        )
        for index, move in enumerate(found[:limit], start=1)
    )


class RapfiAnalyzer:
    """Persistent Rapfi adapter using its Piskvork/Yixin text protocol."""

    name = "Rapfi"

    def __init__(self, config: RapfiConfig) -> None:
        self.config = config
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._warmed = False

    @property
    def available(self) -> bool:
        return self.config.executable.is_file()

    def analyze(self, board: BoardState, limit: int = 3) -> AnalysisResult:
        if not self.available:
            raise FileNotFoundError(f"Rapfi executable not found: {self.config.executable}")
        if board.size != 15:
            raise ValueError("This MVP configures Rapfi for 15x15 only.")
        if not board.is_count_legal():
            raise ValueError("Board counts are invalid.")
        if board.is_terminal():
            return AnalysisResult(board=board, candidates=(), engine_name=self.name)

        with self._lock:
            self._ensure_started()
            self._discard_pending_messages()
            side = board.side_to_move()
            commands = ["INFO SHOW_DETAIL 3", "YXBOARD"]
            for y in range(board.size):
                for x in range(board.size):
                    stone = board.at(x, y)
                    if stone is Stone.EMPTY:
                        continue
                    color = 1 if stone is side else 2
                    commands.append(f"{x},{y},{color}")
            commands.extend(["DONE", f"YXNBEST {limit}"])
            self._write_commands(commands)

            extra_wait = self.config.startup_timeout_s if not self._warmed else 3.0
            deadline = time.monotonic() + (self.config.time_ms / 1000) + extra_wait
            output: list[str] = []
            final_move_seen = False
            while time.monotonic() < deadline:
                remaining = max(deadline - time.monotonic(), 0.01)
                try:
                    line = self._lines.get(timeout=remaining)
                except queue.Empty:
                    break
                if line is None:
                    self.close()
                    break
                output.append(line)
                if re.fullmatch(r"\d{1,2},\d{1,2}", line.strip()):
                    final_move_seen = True
                    break

            raw_output = "\n".join(output)
            candidates = parse_rapfi_output(raw_output, board, limit=limit)
            if not final_move_seen and not candidates:
                self.close()
                raise EngineProtocolError(
                    "Rapfi returned no candidate before the timeout. Output:\n" + raw_output[-2000:]
                )
            engine_name = self.name
            if len(candidates) < limit:
                candidates = self._fill_with_tactical_alternatives(board, candidates, limit)
                engine_name = "Rapfi + tactical alternatives"
            self._warmed = True
            return AnalysisResult(
                board=board,
                candidates=candidates,
                engine_name=engine_name,
                raw_output=raw_output,
            )

    def close(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            self._reader_thread = None
            self._lines = queue.Queue()
            self._warmed = False
            if process is None:
                return
            try:
                if process.stdin:
                    process.stdin.write("END\n")
                    process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self.close()
        process = subprocess.Popen(
            [str(self.config.executable)],
            cwd=str(self.config.executable.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdin is None or process.stdout is None:
            process.kill()
            raise EngineProtocolError("Could not open Rapfi pipes.")

        self._process = process
        self._lines = queue.Queue()

        def reader() -> None:
            try:
                for response in iter(process.stdout.readline, ""):
                    self._lines.put(response.rstrip("\r\n"))
            finally:
                self._lines.put(None)

        self._reader_thread = threading.Thread(target=reader, daemon=True)
        self._reader_thread.start()
        self._write_commands(["START 15"])

        deadline = time.monotonic() + self.config.startup_timeout_s
        startup_output: list[str] = []
        while time.monotonic() < deadline:
            try:
                line = self._lines.get(timeout=max(deadline - time.monotonic(), 0.01))
            except queue.Empty:
                break
            if line is None:
                break
            startup_output.append(line)
            if line.strip() == "OK":
                self._write_commands(
                    [
                        "INFO RULE 0",
                        f"INFO TIMEOUT_TURN {self.config.time_ms}",
                        f"INFO THREAD_NUM {self.config.threads}",
                        f"INFO HASH_SIZE {self.config.hash_kib}",
                        "YXSHOWINFO",
                    ]
                )
                return

        self.close()
        raise EngineProtocolError(
            "Rapfi did not finish startup. Output:\n" + "\n".join(startup_output)[-2000:]
        )

    def _write_commands(self, commands: list[str]) -> None:
        if self._process is None or self._process.stdin is None:
            raise EngineProtocolError("Rapfi process is not running.")
        try:
            self._process.stdin.write("\n".join(commands) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            self.close()
            raise EngineProtocolError("Rapfi process closed its input pipe.") from error

    def _discard_pending_messages(self) -> None:
        while True:
            try:
                self._lines.get_nowait()
            except queue.Empty:
                return

    @staticmethod
    def _fill_with_tactical_alternatives(
        board: BoardState,
        engine_moves: tuple[CandidateMove, ...],
        limit: int,
    ) -> tuple[CandidateMove, ...]:
        merged = list(engine_moves)
        seen = {(move.x, move.y) for move in merged}
        for move in HeuristicAnalyzer().analyze(board, limit=limit + len(merged)).candidates:
            if (move.x, move.y) not in seen:
                merged.append(move)
                seen.add((move.x, move.y))
            if len(merged) >= limit:
                break
        return tuple(
            CandidateMove(
                x=move.x,
                y=move.y,
                rank=index,
                score=move.score,
                proof=move.proof,
                principal_variation=move.principal_variation,
            )
            for index, move in enumerate(merged[:limit], start=1)
        )
