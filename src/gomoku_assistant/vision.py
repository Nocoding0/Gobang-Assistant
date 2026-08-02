from __future__ import annotations

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
    schema_version: int = 2
    source_width: int | None = None
    source_height: int | None = None
    window_title: str | None = None
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
            "schema_version": self.schema_version,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "window_title": self.window_title,
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
            schema_version=int(data.get("schema_version", 1)),
            source_width=(
                int(data["source_width"]) if data.get("source_width") is not None else None
            ),
            source_height=(
                int(data["source_height"]) if data.get("source_height") is not None else None
            ),
            window_title=(
                str(data["window_title"]) if data.get("window_title") is not None else None
            ),
            warp_size=int(data.get("warp_size", DEFAULT_WARP_SIZE)),
            black_gray_max=float(data.get("black_gray_max", 82.0)),
            black_fraction_min=float(data.get("black_fraction_min", 0.46)),
            white_gray_min=float(data.get("white_gray_min", 182.0)),
            white_fraction_min=float(data.get("white_fraction_min", 0.42)),
            white_saturation_max=float(data.get("white_saturation_max", 65.0)),
        )

    def matches_source_shape(
        self, source_width: int, source_height: int, window_title: str | None
    ) -> bool:
        if self.source_width is None or self.source_height is None:
            return False
        if (source_width, source_height) != (self.source_width, self.source_height):
            return False
        return self.window_title is None or self.window_title == window_title

    def matches_source(self, frame_bgr: np.ndarray, window_title: str | None) -> bool:
        return self.matches_source_shape(
            frame_bgr.shape[1], frame_bgr.shape[0], window_title
        )


@dataclass(frozen=True)
class RecognitionResult:
    board: BoardState
    confidence: float
    cell_confidences: tuple[float, ...]
    board_visible: bool
    grid_score: float
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
    grid_score = _grid_score(gray, spacing, profile.board_size)
    board_visible = grid_score >= 0.35
    radius = max(int(spacing * 0.36), 7)
    inner_radius = max(int(spacing * 0.16), 3)
    stone_radius = max(int(spacing * 0.30), 6)

    cells: list[Stone] = []
    confidences: list[float] = []
    for y in range(profile.board_size):
        for x in range(profile.board_size):
            center_x = round(x * spacing)
            center_y = round(y * spacing)
            gray_patch, hsv_patch, distances = _circular_patch(
                gray, hsv, center_x, center_y, radius
            )
            inner_mask = distances <= inner_radius**2
            outer_mask = (distances >= inner_radius**2) & (distances <= stone_radius**2)
            if not np.any(outer_mask):
                cells.append(Stone.EMPTY)
                confidences.append(0.0)
                continue

            dark_mask = gray_patch <= profile.black_gray_max
            white_mask = (gray_patch >= profile.white_gray_min) & (
                hsv_patch[..., 1] <= profile.white_saturation_max
            )
            dark_inner = float(np.mean(dark_mask[inner_mask])) if np.any(inner_mask) else 0.0
            dark_outer = float(np.mean(dark_mask[outer_mask]))
            white_inner = float(np.mean(white_mask[inner_mask])) if np.any(inner_mask) else 0.0
            white_outer = float(np.mean(white_mask[outer_mask]))
            black_round = _centered_round_mask(dark_mask & (distances <= stone_radius**2))
            white_round = _centered_round_mask(white_mask & (distances <= stone_radius**2))

            # The outer ring keeps black stones detectable when the game draws a
            # colored last-move marker in the center.
            if (dark_inner >= profile.black_fraction_min or dark_outer >= 0.28) and black_round:
                cells.append(Stone.BLACK)
                confidences.append(
                    min(1.0, max(dark_inner / profile.black_fraction_min, dark_outer / 0.28))
                )
            elif (
                white_inner >= profile.white_fraction_min
                or white_outer >= 0.36
            ) and white_round:
                cells.append(Stone.WHITE)
                confidences.append(
                    min(1.0, max(white_inner / profile.white_fraction_min, white_outer / 0.36))
                )
            else:
                cells.append(Stone.EMPTY)
                confidence = max(
                    0.0,
                    1.0
                    - max(
                        dark_inner / profile.black_fraction_min,
                        dark_outer / 0.28,
                        white_inner / profile.white_fraction_min,
                        white_outer / 0.36,
                    ),
                )
                confidences.append(confidence)

    return RecognitionResult(
        board=BoardState(size=profile.board_size, cells=tuple(cells)),
        confidence=float(np.mean(confidences)),
        cell_confidences=tuple(confidences),
        board_visible=board_visible,
        grid_score=grid_score,
        warped=warped,
    )


def _circular_patch(
    gray: np.ndarray,
    hsv: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
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
    return gray_patch, hsv_patch, distance_sq


def _centered_round_mask(mask: np.ndarray) -> bool:
    ys, xs = np.nonzero(mask)
    if len(xs) < 16:
        return False
    center_x = (mask.shape[1] - 1) / 2
    center_y = (mask.shape[0] - 1) / 2
    radius = min(mask.shape) / 2
    if np.hypot(xs.mean() - center_x, ys.mean() - center_y) > radius * 0.30:
        return False
    angles = np.arctan2(ys - center_y, xs - center_x)
    sectors = np.unique(np.floor((angles + np.pi) / (2 * np.pi) * 8).astype(int))
    return len(sectors) >= 5


def _grid_score(gray: np.ndarray, spacing: float, board_size: int) -> float:
    """Score expected grid-line contrast for a correctly calibrated board."""

    contrasts: list[float] = []
    line_half_width = max(1, round(spacing * 0.025))
    side_offset = max(3, round(spacing * 0.11))
    for index in range(board_size):
        point = round(index * spacing)
        left = gray[
            :,
            max(0, point - side_offset - line_half_width) : max(
                0, point - side_offset + line_half_width + 1
            ),
        ]
        line = gray[:, max(0, point - line_half_width) : point + line_half_width + 1]
        right = gray[
            :,
            point + side_offset - line_half_width : point + side_offset + line_half_width + 1,
        ]
        if left.size and line.size and right.size:
            contrasts.append(float((left.mean() + right.mean()) / 2 - line.mean()))

        top = gray[
            max(0, point - side_offset - line_half_width) : max(
                0, point - side_offset + line_half_width + 1
            ),
            :,
        ]
        line = gray[max(0, point - line_half_width) : point + line_half_width + 1, :]
        bottom = gray[
            point + side_offset - line_half_width : point + side_offset + line_half_width + 1,
            :,
        ]
        if top.size and line.size and bottom.size:
            contrasts.append(float((top.mean() + bottom.mean()) / 2 - line.mean()))
    if not contrasts:
        return 0.0
    return float(np.clip(np.median(contrasts) / 18.0, 0.0, 1.0))


class StableStateTracker:
    """Commits only repeated frames that form a legal new board state."""

    def __init__(
        self,
        required_frames: int = 3,
        min_confidence: float = 0.70,
        reset_frames: int = 5,
    ) -> None:
        if required_frames < 1:
            raise ValueError("required_frames must be positive")
        self.required_frames = required_frames
        self.min_confidence = min_confidence
        self.reset_frames = reset_frames
        self._samples: list[BoardState] = []
        self._reset_samples: list[BoardState] = []
        self.committed: BoardState | None = None

    def reset(self) -> None:
        self._samples.clear()
        self._reset_samples.clear()
        self.committed = None

    def observe(self, result: RecognitionResult) -> tuple[TransitionResult, BoardState | None]:
        if not result.board_visible:
            self._samples.clear()
            self._reset_samples.clear()
            return TransitionResult(False, False, "board grid is not visible"), None
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
            self._reset_samples.clear()
            self.committed = candidate
            return transition, candidate
        if (
            self.committed is not None
            and candidate.is_count_legal()
            and sum(candidate.counts()) < sum(self.committed.counts())
        ):
            self._reset_samples.append(candidate)
            self._reset_samples = self._reset_samples[-self.reset_frames :]
            if len(self._reset_samples) == self.reset_frames and len(
                set(sample.cells for sample in self._reset_samples)
            ) == 1:
                self.committed = candidate
                return TransitionResult(True, True, "new game detected", sum(candidate.counts())), candidate
        return transition, None
