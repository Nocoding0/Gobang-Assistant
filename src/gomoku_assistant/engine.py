from __future__ import annotations

import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .analysis import AnalysisResult, CandidateMove, HeuristicAnalyzer, ProofStatus, SearchStats
from .domain import BoardState, Stone


COORDINATE_RE = re.compile(r"(?<!\d)(\d{1,2}),(\d{1,2})(?!\d)")
ALGEBRAIC_COORDINATE_RE = re.compile(r"(?<![A-Z])([A-O])(1[0-5]|[1-9])(?!\d)")
MULTIPV_RE = re.compile(r"^MESSAGE\s+\((\d+)\)\s+([+-]?(?:M\d+|\d+))")
FINAL_EVAL_RE = re.compile(r"^MESSAGE\s+Depth\b.*\|\s+Eval\s+([+-]?(?:M\d+|\d+))\s+\|")
REALTIME_BEST_RE = re.compile(r"^MESSAGE\s+REALTIME\s+BEST\s+(\d{1,2}),\s*(\d{1,2})\b")
FINAL_SUMMARY_RE = re.compile(
    r"^MESSAGE\s+Depth\s+([^|]+?)\s+\|\s+Eval\s+[+-]?(?:M\d+|\d+)\s+\|\s+Time\s+(\d+)ms\b"
)

MAX_RAPFI_SEARCH_TIME_MS = 15_000
MAX_TOTAL_ANALYSIS_TIME_MS = 17_000


class EngineProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class RapfiConfig:
    executable: Path
    time_ms: int = 8_000
    threads: int = 8
    hash_kib: int = 512 * 1024
    startup_timeout_s: float = 25.0

    def __post_init__(self) -> None:
        if not 1 <= self.time_ms <= MAX_RAPFI_SEARCH_TIME_MS:
            raise ValueError(
                f"Rapfi search time must be between 1 and {MAX_RAPFI_SEARCH_TIME_MS} ms."
            )
        if self.threads < 1:
            raise ValueError("Rapfi thread count must be positive.")
        if self.hash_kib < 128 * 1024:
            raise ValueError("Rapfi hash must be at least 128 MiB.")


def parse_rapfi_output(
    output: str, board: BoardState, limit: int = 3
) -> tuple[CandidateMove, ...]:
    """Extract ranked candidate moves from Rapfi Yixin detail messages."""

    ranked: dict[int, CandidateMove] = {}
    fallback_point: tuple[int, int] | None = None
    fallback_variation: tuple[tuple[int, int], ...] = ()
    realtime_point: tuple[int, int] | None = None
    realtime_variation: tuple[tuple[int, int], ...] = ()
    final_score: int | None = None
    final_proof = ProofStatus.HEURISTIC
    has_final_evaluation = False

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line and re.fullmatch(r"\d{1,2},\d{1,2}", line):
            fallback_point = tuple(map(int, line.split(",")))
            fallback_variation = (fallback_point,)
        realtime_match = REALTIME_BEST_RE.match(line)
        if realtime_match:
            realtime_point = (int(realtime_match.group(1)), int(realtime_match.group(2)))
        final_eval = FINAL_EVAL_RE.match(line)
        if final_eval:
            final_score, final_proof = _parse_rapfi_score(final_eval.group(1))
            has_final_evaluation = True
            summary_variation = _parse_output_coordinates(line)
            if summary_variation:
                realtime_variation = summary_variation

        match = MULTIPV_RE.match(line)
        coordinates = _parse_output_coordinates(line)
        if not match or not coordinates:
            continue
        rank = int(match.group(1))
        x, y = coordinates[0]
        if not board.in_bounds(x, y) or board.at(x, y) is not Stone.EMPTY:
            continue
        score, proof = _parse_rapfi_score(match.group(2))
        ranked[rank] = CandidateMove(
            x=x,
            y=y,
            rank=rank,
            score=score,
            proof=proof,
            principal_variation=coordinates,
        )

    if fallback_point is None:
        fallback_point = realtime_point
        fallback_variation = realtime_variation or ((realtime_point,) if realtime_point else ())

    if fallback_point:
        x, y = fallback_point
        if board.in_bounds(x, y) and board.at(x, y) is Stone.EMPTY:
            current = ranked.get(1)
            proof = (
                ProofStatus.WIN_IN_ONE
                if board.place(x, y).winner() is board.side_to_move()
                else (
                    final_proof
                    if has_final_evaluation
                    else current.proof if current is not None else ProofStatus.HEURISTIC
                )
            )
            ranked[1] = CandidateMove(
                x=x,
                y=y,
                rank=1,
                score=(
                    final_score
                    if has_final_evaluation
                    else current.score if current is not None else None
                ),
                proof=proof,
                principal_variation=(
                    current.principal_variation
                    if current is not None and (current.x, current.y) == fallback_point
                    else fallback_variation or ((x, y),)
                ),
            )

    return tuple(
        CandidateMove(
            x=move.x,
            y=move.y,
            rank=index,
            score=move.score,
            proof=move.proof,
            principal_variation=move.principal_variation,
        )
        for index, move in enumerate(
            (ranked[rank] for rank in sorted(ranked)[:limit]), start=1
        )
    )


def _parse_output_coordinates(line: str) -> tuple[tuple[int, int], ...]:
    coordinate_text = line.rsplit("|", 1)[-1] if line.startswith("MESSAGE ") else line
    numeric = tuple(
        (int(x), int(y)) for x, y in COORDINATE_RE.findall(coordinate_text)
    )
    if numeric:
        return numeric
    return tuple(
        (ord(column) - ord("A"), int(row) - 1)
        for column, row in ALGEBRAIC_COORDINATE_RE.findall(coordinate_text.upper())
    )


def _parse_rapfi_score(value: str) -> tuple[int | None, ProofStatus]:
    if "M" in value:
        return (
            None,
            ProofStatus.FORCED_WIN if not value.startswith("-") else ProofStatus.FORCED_LOSS,
        )
    score = int(value)
    if score >= 29_000:
        return score, ProofStatus.FORCED_WIN
    if score <= -29_000:
        return score, ProofStatus.FORCED_LOSS
    return score, ProofStatus.HEURISTIC


def parse_rapfi_search_summary(output: str) -> tuple[str | None, int | None]:
    """Return the final iterative-deepening depth and engine-reported time."""

    depth: str | None = None
    engine_time_ms: int | None = None
    for raw_line in output.splitlines():
        match = FINAL_SUMMARY_RE.match(raw_line.strip())
        if match:
            depth = match.group(1).strip()
            engine_time_ms = int(match.group(2))
    return depth, engine_time_ms


def _build_rapfi_search_commands(board: BoardState) -> list[str]:
    """Build a board request with one full-strength Rapfi principal variation."""

    side = board.side_to_move()
    commands = ["INFO SHOW_DETAIL 3", "YXBOARD"]
    for y in range(board.size):
        for x in range(board.size):
            stone = board.at(x, y)
            if stone is Stone.EMPTY:
                continue
            color = 1 if stone is side else 2
            commands.append(f"{x},{y},{color}")
    commands.extend(["DONE", "YXNBEST 1"])
    return commands


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
        self._active_timeout_ms: int | None = None

    @property
    def available(self) -> bool:
        return self.config.executable.is_file()

    def warm_up(self, timeout_ms: int = MAX_TOTAL_ANALYSIS_TIME_MS) -> None:
        """Start and configure Rapfi before the player is waiting for a move."""

        deadline = time.monotonic() + min(timeout_ms, MAX_TOTAL_ANALYSIS_TIME_MS) / 1000
        with self._lock:
            self._ensure_started(deadline)

    def analyze(
        self,
        board: BoardState,
        limit: int = 3,
        *,
        time_ms: int | None = None,
        total_timeout_ms: int = MAX_TOTAL_ANALYSIS_TIME_MS,
    ) -> AnalysisResult:
        requested_time_ms = self.config.time_ms if time_ms is None else time_ms
        if not 1 <= requested_time_ms <= MAX_RAPFI_SEARCH_TIME_MS:
            raise ValueError(
                f"Rapfi search time must be between 1 and {MAX_RAPFI_SEARCH_TIME_MS} ms."
            )
        hard_timeout_ms = min(total_timeout_ms, MAX_TOTAL_ANALYSIS_TIME_MS)
        if hard_timeout_ms < requested_time_ms:
            raise ValueError("Total analysis time cannot be shorter than the engine search time.")
        if not self.available:
            raise FileNotFoundError(f"Rapfi executable not found: {self.config.executable}")
        if board.size != 15:
            raise ValueError("This MVP configures Rapfi for 15x15 only.")
        if not board.is_count_legal():
            raise ValueError("Board counts are invalid.")
        if board.is_terminal():
            return AnalysisResult(board=board, candidates=(), engine_name=self.name)

        started_at = time.monotonic()
        hard_deadline = started_at + hard_timeout_ms / 1000

        with self._lock:
            self._ensure_started(hard_deadline)
            if time.monotonic() >= hard_deadline:
                raise EngineProtocolError("Rapfi startup used the full 17 second analysis budget.")
            self._discard_pending_messages()
            self._set_search_timeout(requested_time_ms)
            self._write_commands(_build_rapfi_search_commands(board))

            deadline = min(hard_deadline, time.monotonic() + requested_time_ms / 1000)
            output: list[str] = []
            final_move_seen = False
            while time.monotonic() < deadline:
                remaining = max(deadline - time.monotonic(), 0.01)
                try:
                    line = self._lines.get(timeout=remaining)
                except queue.Empty:
                    break
                if line is None:
                    self.close(wait_timeout_s=0)
                    break
                output.append(line)
                if re.fullmatch(r"\d{1,2},\d{1,2}", line.strip()):
                    final_move_seen = True
                    break

            raw_output = "\n".join(output)
            candidates = parse_rapfi_output(raw_output, board, limit=1)
            if not final_move_seen and not candidates:
                raise EngineProtocolError(
                    "Rapfi returned no candidate before the timeout. Output:\n" + raw_output[-2000:]
                )
            engine_name = self.name
            if len(candidates) < limit:
                candidates = self._fill_with_tactical_alternatives(board, candidates, limit)
                engine_name = "Rapfi + tactical alternatives"
            self._warmed = True
            depth, engine_time_ms = parse_rapfi_search_summary(raw_output)
            elapsed_ms = int((time.monotonic() - started_at) * 1000)
            return AnalysisResult(
                board=board,
                candidates=candidates,
                engine_name=engine_name,
                raw_output=raw_output,
                search_stats=SearchStats(
                    requested_time_ms=requested_time_ms,
                    elapsed_ms=elapsed_ms,
                    engine_time_ms=engine_time_ms,
                    depth=depth,
                    threads=self.config.threads,
                    hash_kib=self.config.hash_kib,
                ),
            )

    def close(self, wait_timeout_s: float = 2.0) -> None:
        with self._lock:
            process = self._process
            self._process = None
            self._reader_thread = None
            self._lines = queue.Queue()
            self._warmed = False
            self._active_timeout_ms = None
            if process is None:
                return
            try:
                if process.stdin:
                    process.stdin.write("END\n")
                    process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            if wait_timeout_s <= 0:
                process.kill()
                return
            try:
                process.wait(timeout=wait_timeout_s)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=wait_timeout_s)
                except subprocess.TimeoutExpired:
                    pass

    def _ensure_started(self, deadline: float) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self.close(wait_timeout_s=0)
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

        startup_deadline = min(deadline, time.monotonic() + self.config.startup_timeout_s)
        startup_output: list[str] = []
        while time.monotonic() < startup_deadline:
            try:
                line = self._lines.get(timeout=max(startup_deadline - time.monotonic(), 0.01))
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
                self._active_timeout_ms = self.config.time_ms
                return

        self.close(wait_timeout_s=0)
        raise EngineProtocolError(
            "Rapfi did not finish startup. Output:\n" + "\n".join(startup_output)[-2000:]
        )

    def _set_search_timeout(self, time_ms: int) -> None:
        if self._active_timeout_ms == time_ms:
            return
        self._write_commands([f"INFO TIMEOUT_TURN {time_ms}"])
        self._active_timeout_ms = time_ms

    def _write_commands(self, commands: list[str]) -> None:
        if self._process is None or self._process.stdin is None:
            raise EngineProtocolError("Rapfi process is not running.")
        try:
            self._process.stdin.write("\n".join(commands) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            self.close(wait_timeout_s=0)
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
