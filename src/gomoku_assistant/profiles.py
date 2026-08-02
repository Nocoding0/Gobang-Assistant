from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from .vision import BoardProfile

if TYPE_CHECKING:
    from .capture import WindowInfo


class ProfileStore:
    """Persists a calibration per window title and physical capture size."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def load_for(self, window: "WindowInfo") -> BoardProfile | None:
        profile_path = self._path_for(window.title, window.width, window.height)
        if profile_path.is_file():
            return self._load(profile_path)

        legacy = self.directory / "default.json"
        if legacy.is_file():
            profile = self._load(legacy)
            if profile and profile.matches_source_shape(
                window.width, window.height, window.title
            ):
                self.save(profile)
                return profile
        return None

    def save(self, profile: BoardProfile) -> Path:
        if (
            profile.source_width is None
            or profile.source_height is None
            or profile.window_title is None
        ):
            raise ValueError("Profiles need a window title and capture dimensions.")
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self._path_for(
            profile.window_title, profile.source_width, profile.source_height
        )
        target.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
        return target

    def _path_for(self, title: str, width: int, height: int) -> Path:
        identity = f"{title}\0{width}x{height}".encode("utf-8")
        digest = hashlib.sha256(identity).hexdigest()[:12]
        return self.directory / f"profile-{digest}.json"

    @staticmethod
    def _load(path: Path) -> BoardProfile | None:
        try:
            return BoardProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
