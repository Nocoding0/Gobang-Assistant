from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .analysis import AnalysisResult
@dataclass(frozen=True)
class SessionEntry:
    at_utc: str
    board: tuple[int, ...]
    size: int
    engine: str
    candidates: tuple[dict[str, object], ...]


class SessionLogger:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.entries: list[SessionEntry] = []

    def append(self, result: AnalysisResult) -> None:
        self.entries.append(
            SessionEntry(
                at_utc=datetime.now(timezone.utc).isoformat(),
                board=tuple(int(cell) for cell in result.board.cells),
                size=result.board.size,
                engine=result.engine_name,
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
            )
        )

    def save(self) -> Path | None:
        if not self.entries:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        filename = datetime.now().strftime("%Y%m%d-%H%M%S") + ".json"
        target = self.directory / filename
        target.write_text(
            json.dumps([asdict(entry) for entry in self.entries], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target
