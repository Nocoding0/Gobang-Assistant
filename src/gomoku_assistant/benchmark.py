from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .domain import BoardState, Stone
from .engine import MAX_RAPFI_SEARCH_TIME_MS, RapfiAnalyzer, RapfiConfig


@dataclass(frozen=True)
class LoggedPosition:
    source: str
    board: BoardState
    baseline_move: tuple[int, int] | None


def load_logged_positions(directory: Path) -> tuple[LoggedPosition, ...]:
    """Read both legacy session arrays and schema-v2 session envelopes."""

    positions: list[LoggedPosition] = []
    for path in sorted(directory.glob("*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        entries = payload if isinstance(payload, list) else payload.get("analyses", [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            board = _board_from_entry(entry)
            if board is None or board.is_terminal() or not board.is_count_legal():
                continue
            baseline_move = _rank_one_move(entry.get("candidates"))
            positions.append(
                LoggedPosition(path.name, board, baseline_move)
            )
    return _unique_positions(positions)


def benchmark_positions(
    positions: Iterable[LoggedPosition],
    analyzer: RapfiAnalyzer,
    black_time_ms: int,
    white_time_ms: int,
    maximum: int,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for position in list(positions)[:maximum]:
        board = position.board
        budget_ms = black_time_ms if board.side_to_move() is Stone.BLACK else white_time_ms
        result = analyzer.analyze(board, time_ms=budget_ms)
        move = result.candidates[0] if result.candidates else None
        target_move = (move.x, move.y) if move is not None else None
        records.append(
            {
                "source": position.source,
                "move_number": sum(board.counts()),
                "side_to_move": board.side_to_move().name.lower(),
                "baseline_move": position.baseline_move,
                "target_move": target_move,
                "changed": (
                    position.baseline_move is not None
                    and target_move is not None
                    and position.baseline_move != target_move
                ),
                "proof": move.proof.value if move is not None else None,
                "score": move.score if move is not None else None,
                "search": asdict(result.search_stats) if result.search_stats else None,
            }
        )
    changed = sum(record["changed"] is True for record in records)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "positions": records,
        "summary": {
            "analyzed": len(records),
            "changed_from_logged_rank_one": changed,
            "unchanged_or_no_baseline": len(records) - changed,
        },
    }


def _board_from_entry(entry: dict[str, object]) -> BoardState | None:
    board_values = entry.get("board")
    size = entry.get("size")
    if not isinstance(board_values, list | tuple) or not isinstance(size, int):
        return None
    try:
        return BoardState(size=size, cells=tuple(Stone(value) for value in board_values))
    except (TypeError, ValueError):
        return None


def _rank_one_move(candidates: object) -> tuple[int, int] | None:
    if not isinstance(candidates, list | tuple):
        return None
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("rank") != 1:
            continue
        x, y = candidate.get("x"), candidate.get("y")
        if isinstance(x, int) and isinstance(y, int):
            return x, y
    return None


def _unique_positions(positions: Iterable[LoggedPosition]) -> tuple[LoggedPosition, ...]:
    unique: list[LoggedPosition] = []
    seen: set[tuple[int, ...]] = set()
    for position in positions:
        key = tuple(int(cell) for cell in position.board.cells)
        if key in seen:
            continue
        seen.add(key)
        unique.append(position)
    return tuple(unique)


def _time_argument(value: str) -> int:
    seconds = int(value)
    if not 1 <= seconds <= MAX_RAPFI_SEARCH_TIME_MS // 1000:
        raise argparse.ArgumentTypeError("time must be between 1 and 15 seconds")
    return seconds * 1000


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare logged rank-one moves against the current bounded Rapfi settings."
    )
    parser.add_argument("--sessions", type=Path, default=Path("sessions"))
    parser.add_argument("--rapfi", type=Path, required=True)
    parser.add_argument("--black-seconds", type=_time_argument, default=8_000)
    parser.add_argument("--white-seconds", type=_time_argument, default=15_000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--hash-mb", type=int, choices=(128, 256, 512, 1024), default=512)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.max_positions < 1:
        parser.error("--max-positions must be positive")
    positions = load_logged_positions(args.sessions)
    if not positions:
        parser.error("no legal analyzed positions found in the session directory")

    analyzer = RapfiAnalyzer(
        RapfiConfig(
            executable=args.rapfi,
            time_ms=MAX_RAPFI_SEARCH_TIME_MS,
            threads=args.threads,
            hash_kib=args.hash_mb * 1024,
        )
    )
    try:
        analyzer.warm_up()
        report = benchmark_positions(
            positions,
            analyzer,
            args.black_seconds,
            args.white_seconds,
            args.max_positions,
        )
    finally:
        analyzer.close(wait_timeout_s=0)

    output = args.output or Path("benchmarks") / (
        datetime.now().strftime("%Y%m%d-%H%M%S") + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = report["summary"]
    print(
        f"Analyzed {summary['analyzed']} positions; "
        f"{summary['changed_from_logged_rank_one']} rank-one moves changed.\n"
        f"Report: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
