from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

from .domain import BoardState, Stone, TransitionResult, validate_transition


DEFAULT_WARP_SIZE = 840


@dataclass(frozen=True)
class BoardProfile:
    board_size: int
    corners: tuple[tuple[float, float], ...]
    warp_size: int = DEFAULT_WARP_SIZE
    black_gray_max: float = 82.0
    black_fraction_min: float = 0.46
    white_gray_min: float = 182.0
    white_fraction_min: float = 0.42
    white_saturation_max: float = 65.0

    def __post_init__(self) -> None:
        if self.board_size != 15:
            raise ValueError("This MVP supports 15x15 profiles only.")
        if len(self.corners) != 4:
            raise ValueError("A profile requires exactly four corners.")
        if self.warp_size % (self.board_size - 1) != 0:
            raise ValueError("Warp size must divide evenly into grid intervals.")

    @property
    def spacing(self) -> float:
        return self.warp_size / (self.board_size - 1)

    def to_dict(self) -> dict[str, object]:
        return {
            "board_size": self.board_size,
            "corners": [list(point) for point in self.corners],
            "warp_size": self.warp_size,
            "black_gray_max": self.black_gray_max,
            "black_fraction_min": self.black_fraction_min,
            "white_gray_min": self.white_gray_min,
            "white_fraction_min": self.white_fraction_min,
            "white_saturation_max": self.white_saturation_max,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "BoardProfile":
        return cls(
            board_size=int(data["board_size"]),
            corners=tuple(tuple(map(float, point)) for point in data["corners"]),  # type: ignore[arg-type]
            warp_size=int(data.get("warp_size", DEFAULT_WARP_SIZE)),
            black_gray_max=float(data.get("black_gray_max", 82.0)),
            black_fraction_min=float(data.get("black_fraction_min", 0.46)),
            white_gray_min=float(data.get("white_gray_min", 182.0)),
            white_fraction_min=float(data.get("white_fraction_min", 0.42)),
            white_saturation_max=float(data.get("white_saturation_max", 65.0)),
        )


@dataclass(frozen=True)
class RecognitionResult:
    board: BoardState
    confidence: float
    cell_confidences: tuple[float, ...]
    warped: np.ndarray = field(repr=False, compare=False)


def order_corners(points: Iterable[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """Return corners in top-left, top-right, bottom-right, bottom-left order."""

    value = np.array(tuple(points), dtype=np.float32)
    if value.shape != (4, 2):
        raise ValueError("Exactly four 2D points are required.")
    sums = value.sum(axis=1)
    diffs = np.diff(value, axis=1).reshape(-1)
    top_left = value[np.argmin(sums)]
    bottom_right = value[np.argmax(sums)]
    top_right = value[np.argmin(diffs)]
    bottom_left = value[np.argmax(diffs)]
    return tuple((float(point[0]), float(point[1])) for point in (top_left, top_right, bottom_right, bottom_left))


def warp_board(frame_bgr: np.ndarray, profile: BoardProfile) -> np.ndarray:
    source = np.array(profile.corners, dtype=np.float32)
    edge = profile.warp_size
    destination = np.array(
        [(0, 0), (edge, 0), (edge, edge), (0, edge)],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(frame_bgr, transform, (edge + 1, edge + 1))


def grid_points_in_source(profile: BoardProfile) -> tuple[tuple[float, float], ...]:
    source = np.array(profile.corners, dtype=np.float32)
    edge = profile.warp_size
    destination = np.array(
        [(0, 0), (edge, 0), (edge, edge), (0, edge)],
        dtype=np.float32,
    )
    inverse = cv2.getPerspectiveTransform(destination, source)
    points = np.array(
        [
            [[x * profile.spacing, y * profile.spacing]]
            for y in range(profile.board_size)
            for x in range(profile.board_size)
        ],
        dtype=np.float32,
    )
    projected = cv2.perspectiveTransform(points, inverse).reshape(-1, 2)
    return tuple((float(x), float(y)) for x, y in projected)


def recognize_frame(frame_bgr: np.ndarray, profile: BoardProfile) -> RecognitionResult:
    warped = warp_board(frame_bgr, profile)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    spacing = profile.spacing
    radius = max(int(spacing * 0.30), 6)
    center_exclusion = max(int(spacing * 0.075), 2)

    cells: list[Stone] = []
    confidences: list[float] = []
    for y in range(profile.board_size):
        for x in range(profile.board_size):
            center_x = round(x * spacing)
            center_y = round(y * spacing)
            gray_patch, hsv_patch, mask = _circular_patch(
                gray, hsv, center_x, center_y, radius, center_exclusion
            )
            masked_gray = gray_patch[mask]
            masked_saturation = hsv_patch[..., 1][mask]
            if not len(masked_gray):
                cells.append(Stone.EMPTY)
                confidences.append(0.0)
                continue

            dark_fraction = float(np.mean(masked_gray <= profile.black_gray_max))
            white_fraction = float(
                np.mean(
                    (masked_gray >= profile.white_gray_min)
                    & (masked_saturation <= profile.white_saturation_max)
                )
            )
            if dark_fraction >= profile.black_fraction_min:
                cells.append(Stone.BLACK)
                confidences.append(min(1.0, dark_fraction / profile.black_fraction_min))
            elif white_fraction >= profile.white_fraction_min:
                cells.append(Stone.WHITE)
                confidences.append(min(1.0, white_fraction / profile.white_fraction_min))
            else:
                cells.append(Stone.EMPTY)
                confidence = max(
                    0.0,
                    1.0
                    - max(
                        dark_fraction / profile.black_fraction_min,
                        white_fraction / profile.white_fraction_min,
                    ),
                )
                confidences.append(confidence)

    return RecognitionResult(
        board=BoardState(size=profile.board_size, cells=tuple(cells)),
        confidence=float(np.mean(confidences)),
        cell_confidences=tuple(confidences),
        warped=warped,
    )


def _circular_patch(
    gray: np.ndarray,
    hsv: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
    center_exclusion: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = gray.shape
    left = max(center_x - radius, 0)
    right = min(center_x + radius + 1, width)
    top = max(center_y - radius, 0)
    bottom = min(center_y + radius + 1, height)
    gray_patch = gray[top:bottom, left:right]
    hsv_patch = hsv[top:bottom, left:right]
    yy, xx = np.ogrid[top:bottom, left:right]
    distance_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
    mask = (distance_sq <= radius**2) & (distance_sq >= center_exclusion**2)
    return gray_patch, hsv_patch, mask


class StableStateTracker:
    """Commits only repeated frames that form a legal new board state."""

    def __init__(self, required_frames: int = 3, min_confidence: float = 0.70) -> None:
        if required_frames < 1:
            raise ValueError("required_frames must be positive")
        self.required_frames = required_frames
        self.min_confidence = min_confidence
        self._samples: list[BoardState] = []
        self.committed: BoardState | None = None

    def reset(self) -> None:
        self._samples.clear()
        self.committed = None

    def observe(self, result: RecognitionResult) -> tuple[TransitionResult, BoardState | None]:
        if result.confidence < self.min_confidence:
            self._samples.clear()
            return TransitionResult(False, False, "recognition confidence is too low"), None

        self._samples.append(result.board)
        self._samples = self._samples[-self.required_frames :]
        if len(self._samples) < self.required_frames:
            return TransitionResult(True, False, "waiting for stable frames"), None
        if len(set(sample.cells for sample in self._samples)) != 1:
            return TransitionResult(True, False, "frames are not stable"), None

        candidate = self._samples[-1]
        transition = validate_transition(self.committed, candidate)
        if transition.valid and transition.changed:
            self.committed = candidate
            return transition, candidate
        return transition, None

