from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .analysis import AnalysisResult
from .domain import BoardState, CorrectionEvent, ObservedMove, Stone


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SessionEntry:
    at_utc: str
    board: tuple[int, ...]
    size: int
    engine: str
    candidates: tuple[dict[str, object], ...]
    game_id: int = 0
    move_number: int = 0
    side_to_move: str = "black"
    search: dict[str, object] | None = None
    recommendation_mode: str = "normal"
    danger_points: tuple[tuple[int, int], ...] = ()
    safe_candidate_count: int = 0
    rejected_moves: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class MoveEntry:
    at_utc: str
    game_id: int
    x: int
    y: int
    stone: str
    number: int | None
    certain: bool
    source: str


@dataclass(frozen=True)
class CorrectionEntry:
    at_utc: str
    game_id: int
    x: int
    y: int
    before: str
    after: str
    vision: str
    action: str


@dataclass(frozen=True)
class GameResult:
    at_utc: str
    game_id: int
    winner: str | None
    result: str
    board: tuple[int, ...]
    size: int


class SessionLogger:
    """Collects replay information without changing any engine decision."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.entries: list[SessionEntry] = []
        self.moves: list[MoveEntry] = []
        self.corrections: list[CorrectionEntry] = []
        self.results: list[GameResult] = []
        self.games: list[dict[str, object]] = []
        self._current_game_id = 0

    @property
    def current_game_id(self) -> int:
        return self._current_game_id

    def start_game(self, my_color: Stone | None = None) -> int:
        self._current_game_id += 1
        self.games.append(
            {
                "id": self._current_game_id,
                "started_at_utc": _utc_now(),
                "my_color": my_color.name.lower() if my_color is not None else None,
            }
        )
        return self._current_game_id

    def record_moves(self, moves: tuple[ObservedMove, ...], source: str) -> None:
        if not moves:
            return
        game_id = self._ensure_game()
        timestamp = _utc_now()
        self.moves.extend(
            MoveEntry(
                at_utc=timestamp,
                game_id=game_id,
                x=move.x,
                y=move.y,
                stone=move.stone.name.lower(),
                number=move.number,
                certain=move.certain,
                source=source,
            )
            for move in moves
        )

    def record_correction(self, event: CorrectionEvent) -> None:
        self.corrections.append(
            CorrectionEntry(
                at_utc=_utc_now(),
                game_id=self._ensure_game(),
                x=event.x,
                y=event.y,
                before=event.before.name.lower(),
                after=event.after.name.lower(),
                vision=event.vision.name.lower(),
                action=event.action,
            )
        )

    def append(self, result: AnalysisResult) -> None:
        stats = result.search_stats
        self.entries.append(
            SessionEntry(
                at_utc=_utc_now(),
                board=tuple(int(cell) for cell in result.board.cells),
                size=result.board.size,
                engine=result.engine_name,
                game_id=self._ensure_game(),
                move_number=sum(result.board.counts()),
                side_to_move=result.board.side_to_move().name.lower(),
                search=(
                    {
                        "requested_time_ms": stats.requested_time_ms,
                        "elapsed_ms": stats.elapsed_ms,
                        "engine_time_ms": stats.engine_time_ms,
                        "depth": stats.depth,
                        "threads": stats.threads,
                        "hash_kib": stats.hash_kib,
                    }
                    if stats is not None
                    else None
                ),
                candidates=tuple(
                    {
                        "x": move.x,
                        "y": move.y,
                        "rank": move.rank,
                        "score": move.score,
                        "proof": move.proof.value,
                        "pv": list(move.principal_variation),
                    }
                    for move in result.candidates
                ),
                recommendation_mode=result.recommendation_mode.value,
                danger_points=result.danger_points,
                safe_candidate_count=result.safe_candidate_count,
                rejected_moves=tuple(
                    {
                        "x": move.x,
                        "y": move.y,
                        "source": move.source,
                        "reason": move.reason,
                        "danger_points": move.danger_points,
                    }
                    for move in result.rejected_moves
                ),
            )
        )

    def finish_game(self, board: BoardState) -> None:
        game_id = self._ensure_game()
        if any(result.game_id == game_id for result in self.results):
            return
        winner = board.winner()
        self.results.append(
            GameResult(
                at_utc=_utc_now(),
                game_id=game_id,
                winner=winner.name.lower() if winner is not None else None,
                result=("draw" if winner is None else "win"),
                board=tuple(int(cell) for cell in board.cells),
                size=board.size,
            )
        )

    def save(self) -> Path | None:
        if not (self.entries or self.moves or self.corrections or self.results):
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        filename = datetime.now().strftime("%Y%m%d-%H%M%S") + ".json"
        target = self.directory / filename
        target.write_text(
            json.dumps(
                {
                    "schema_version": 4,
                    "games": self.games,
                    "moves": [asdict(move) for move in self.moves],
                    "corrections": [asdict(correction) for correction in self.corrections],
                    "analyses": [asdict(entry) for entry in self.entries],
                    "results": [asdict(result) for result in self.results],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return target

    def _ensure_game(self) -> int:
        if self._current_game_id == 0:
            return self.start_game()
        return self._current_game_id
