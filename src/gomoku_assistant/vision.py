from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

from .domain import BoardState, Stone, TransitionResult, validate_transition


DEFAULT_WARP_SIZE = 840
MARKER_HUE_MIN = 5
MARKER_HUE_MAX = 30
MARKER_SATURATION_MIN = 100
MARKER_VALUE_MIN = 120
MARKER_INNER_FRACTION_MIN = 0.20
MARKER_BLACK_OUTER_MIN = 0.45
MARKER_BLACK_DISK_MIN = 0.38
MARKER_BLACK_LOW_SATURATION_MIN = 0.50
EDGE_BLACK_INNER_MIN = 0.70
EDGE_BLACK_OUTER_MIN = 0.50
EDGE_BLACK_DISK_MIN = 0.72
EDGE_WHITE_INNER_MIN = 0.70
EDGE_WHITE_OUTER_MIN = 0.60
EDGE_WHITE_DISK_MIN = 0.72


@dataclass(frozen=True)
class BoardProfile:
    board_size: int
    corners: tuple[tuple[float, float], ...]
    schema_version: int = 3
    source_width: int | None = None
    source_height: int | None = None
    window_title: str | None = None
    warp_size: int = DEFAULT_WARP_SIZE
    grid_score_baseline: float | None = None
    black_gray_max: float = 82.0
    black_fraction_min: float = 0.46
    white_gray_min: float = 182.0
    white_fraction_min: float = 0.42
    white_saturation_max: float = 65.0
    black_disk_fraction_min: float = 0.55
    white_disk_fraction_min: float = 0.50
    black_saturation_max: float = 65.0
    black_low_saturation_fraction_min: float = 0.55

    def __post_init__(self) -> None:
        if self.board_size != 15:
            raise ValueError("This MVP supports 15x15 profiles only.")
        if len(self.corners) != 4:
            raise ValueError("A profile requires exactly four corners.")
        if self.warp_size % (self.board_size - 1) != 0:
            raise ValueError("Warp size must divide evenly into grid intervals.")
        if not 0 < self.black_disk_fraction_min <= 1:
            raise ValueError("Black disk coverage must be between 0 and 1.")
        if not 0 < self.white_disk_fraction_min <= 1:
            raise ValueError("White disk coverage must be between 0 and 1.")
        if not 0 <= self.black_saturation_max <= 255:
            raise ValueError("Black saturation maximum must be between 0 and 255.")
        if not 0 < self.black_low_saturation_fraction_min <= 1:
            raise ValueError("Black low-saturation coverage must be between 0 and 1.")

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
            "grid_score_baseline": self.grid_score_baseline,
            "black_gray_max": self.black_gray_max,
            "black_fraction_min": self.black_fraction_min,
            "white_gray_min": self.white_gray_min,
            "white_fraction_min": self.white_fraction_min,
            "white_saturation_max": self.white_saturation_max,
            "black_disk_fraction_min": self.black_disk_fraction_min,
            "white_disk_fraction_min": self.white_disk_fraction_min,
            "black_saturation_max": self.black_saturation_max,
            "black_low_saturation_fraction_min": self.black_low_saturation_fraction_min,
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
            grid_score_baseline=(
                float(data["grid_score_baseline"])
                if data.get("grid_score_baseline") is not None
                else None
            ),
            black_gray_max=float(data.get("black_gray_max", 82.0)),
            black_fraction_min=float(data.get("black_fraction_min", 0.46)),
            white_gray_min=float(data.get("white_gray_min", 182.0)),
            white_fraction_min=float(data.get("white_fraction_min", 0.42)),
            white_saturation_max=float(data.get("white_saturation_max", 65.0)),
            black_disk_fraction_min=float(data.get("black_disk_fraction_min", 0.55)),
            white_disk_fraction_min=float(data.get("white_disk_fraction_min", 0.50)),
            black_saturation_max=float(data.get("black_saturation_max", 65.0)),
            black_low_saturation_fraction_min=float(
                data.get("black_low_saturation_fraction_min", 0.55)
            ),
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

    @property
    def grid_visibility_threshold(self) -> float:
        """Return the minimum signal expected for this calibrated board."""

        if self.grid_score_baseline is None:
            return 0.35
        return max(0.08, min(0.85, self.grid_score_baseline * 0.45))


@dataclass(frozen=True)
class RecognitionResult:
    board: BoardState
    confidence: float
    cell_confidences: tuple[float, ...]
    board_visible: bool
    grid_score: float
    warped: np.ndarray = field(repr=False, compare=False)
    obstruction_reason: str | None = None
    cell_evidence: tuple["CellEvidence", ...] = ()
    ambiguous_points: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class CellEvidence:
    """Normalized support for the three possible values of one board cell."""

    black: float
    white: float
    empty: float

    @property
    def stone(self) -> Stone:
        choices = ((self.empty, Stone.EMPTY), (self.black, Stone.BLACK), (self.white, Stone.WHITE))
        return max(choices, key=lambda choice: choice[0])[1]

    @property
    def confidence(self) -> float:
        return max(self.black, self.white, self.empty)

    @property
    def margin(self) -> float:
        values = sorted((self.black, self.white, self.empty), reverse=True)
        return values[0] - values[1]

    def runner_up(self) -> Stone:
        choices = sorted(
            ((self.empty, Stone.EMPTY), (self.black, Stone.BLACK), (self.white, Stone.WHITE)),
            key=lambda choice: choice[0],
            reverse=True,
        )
        return choices[1][1]


@dataclass(frozen=True)
class GridAssessment:
    """Grid-line signal measured in a perspective-normalized board image."""

    vertical_score: float
    horizontal_score: float

    @property
    def score(self) -> float:
        return min(self.vertical_score, self.horizontal_score)


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


def is_valid_board_quadrilateral(
    points: Iterable[tuple[float, float]], frame_shape: tuple[int, ...]
) -> bool:
    """Validate that four clicks describe a usable board-sized quadrilateral."""

    height, width = frame_shape[:2]
    value = np.array(tuple(points), dtype=np.float32)
    if value.shape != (4, 2):
        return False
    if (
        np.any(value[:, 0] < 0)
        or np.any(value[:, 0] >= width)
        or np.any(value[:, 1] < 0)
        or np.any(value[:, 1] >= height)
    ):
        return False

    ordered = np.array(order_corners(value), dtype=np.float32)
    if not cv2.isContourConvex(ordered.reshape(-1, 1, 2)):
        return False
    if cv2.contourArea(ordered) < width * height * 0.008:
        return False

    edges = np.linalg.norm(ordered - np.roll(ordered, -1, axis=0), axis=1)
    if float(edges.min()) < 1.0 or float(edges.min() / edges.max()) < 0.20:
        return False
    return (
        max(edges[0], edges[2]) / min(edges[0], edges[2]) <= 2.5
        and max(edges[1], edges[3]) / min(edges[1], edges[3]) <= 2.5
    )


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
    grid_assessment = assess_grid(warped, spacing, profile.board_size)
    grid_score = grid_assessment.score
    board_visible = grid_score >= profile.grid_visibility_threshold
    obstruction_reason = assess_obstruction(warped, spacing, profile)
    radius = max(int(spacing * 0.52), 10)
    inner_radius = max(int(spacing * 0.16), 3)
    stone_radius = max(int(spacing * 0.30), 6)

    cells: list[Stone] = []
    confidences: list[float] = []
    evidence: list[CellEvidence] = []
    ambiguous_points: list[tuple[int, int]] = []
    for y in range(profile.board_size):
        for x in range(profile.board_size):
            center_x = round(x * spacing)
            center_y = round(y * spacing)
            cell_evidence = _classify_intersection(
                gray,
                hsv,
                profile,
                x,
                y,
                center_x,
                center_y,
                radius,
                inner_radius,
                stone_radius,
            )
            cells.append(cell_evidence.stone)
            confidences.append(cell_evidence.confidence)
            evidence.append(cell_evidence)
            if cell_evidence.margin < 0.20:
                ambiguous_points.append((x, y))

    return RecognitionResult(
        board=BoardState(size=profile.board_size, cells=tuple(cells)),
        confidence=float(np.mean(confidences)),
        cell_confidences=tuple(confidences),
        board_visible=board_visible,
        grid_score=grid_score,
        warped=warped,
        obstruction_reason=obstruction_reason,
        cell_evidence=tuple(evidence),
        ambiguous_points=tuple(ambiguous_points),
    )


def _classify_intersection(
    gray: np.ndarray,
    hsv: np.ndarray,
    profile: BoardProfile,
    x: int,
    y: int,
    center_x: int,
    center_y: int,
    radius: int,
    inner_radius: int,
    stone_radius: int,
) -> CellEvidence:
    """Evaluate a grid point and tolerate a small calibration-center error."""

    base = _classify_at_center(
        gray, hsv, profile, x, y, center_x, center_y, radius, inner_radius, stone_radius
    )
    is_edge = x in (0, profile.board_size - 1) or y in (0, profile.board_size - 1)
    # The normal case is a clear center classification. Searching every one of
    # 225 intersections would make the 250ms observation loop lag noticeably.
    if not is_edge and base.margin >= 0.45:
        return base
    offset = max(1, round(profile.spacing * 0.08))
    best = base
    best_stone_score = max(base.black, base.white)
    for offset_x, offset_y in ((-offset, 0), (offset, 0), (0, -offset), (0, offset)):
        candidate = _classify_at_center(
            gray,
            hsv,
            profile,
            x,
            y,
            center_x + offset_x,
            center_y + offset_y,
            radius,
            inner_radius,
            stone_radius,
        )
        candidate_score = max(candidate.black, candidate.white) - 0.05
        if candidate.stone is not Stone.EMPTY and candidate_score > best_stone_score:
            best = candidate
            best_stone_score = candidate_score
    return best


def _classify_at_center(
    gray: np.ndarray,
    hsv: np.ndarray,
    profile: BoardProfile,
    x: int,
    y: int,
    center_x: int,
    center_y: int,
    radius: int,
    inner_radius: int,
    stone_radius: int,
) -> CellEvidence:
    gray_patch, hsv_patch, distances = _circular_patch(gray, hsv, center_x, center_y, radius)
    inner_mask = distances <= inner_radius**2
    outer_mask = (distances >= inner_radius**2) & (distances <= stone_radius**2)
    disk_mask = distances <= stone_radius**2
    if not np.any(outer_mask) or not np.any(disk_mask):
        return CellEvidence(black=0.0, white=0.0, empty=1.0)

    dark_mask = gray_patch <= profile.black_gray_max
    white_mask = (gray_patch >= profile.white_gray_min) & (
        hsv_patch[..., 1] <= profile.white_saturation_max
    )
    marker_mask = (
        (hsv_patch[..., 0] >= MARKER_HUE_MIN)
        & (hsv_patch[..., 0] <= MARKER_HUE_MAX)
        & (hsv_patch[..., 1] >= MARKER_SATURATION_MIN)
        & (hsv_patch[..., 2] >= MARKER_VALUE_MIN)
    )
    dark_inner = float(np.mean(dark_mask[inner_mask])) if np.any(inner_mask) else 0.0
    dark_outer = float(np.mean(dark_mask[outer_mask]))
    white_inner = float(np.mean(white_mask[inner_mask])) if np.any(inner_mask) else 0.0
    white_outer = float(np.mean(white_mask[outer_mask]))
    dark_disk = float(np.mean(dark_mask[disk_mask]))
    white_disk = float(np.mean(white_mask[disk_mask]))
    marker_inner = float(np.mean(marker_mask[inner_mask])) if np.any(inner_mask) else 0.0
    dark_pixels = dark_mask & disk_mask
    dark_low_saturation = (
        float(np.mean(hsv_patch[..., 1][dark_pixels] <= profile.black_saturation_max))
        if np.any(dark_pixels)
        else 0.0
    )

    background_mask = (distances >= round(stone_radius * 1.22) ** 2) & (distances <= radius**2)
    background_gray = (
        float(np.median(gray_patch[background_mask])) if np.any(background_mask) else float(np.median(gray_patch))
    )
    disk_gray = float(np.median(gray_patch[disk_mask]))
    relative_dark_mask = (gray_patch <= background_gray - 16.0) & disk_mask
    relative_light_mask = (gray_patch >= background_gray + 14.0) & disk_mask
    relative_black_round = _centered_round_mask(relative_dark_mask)
    relative_white_round = _centered_round_mask(relative_light_mask)
    relative_dark_low_saturation = (
        float(np.mean(hsv_patch[..., 1][relative_dark_mask] <= profile.black_saturation_max))
        if np.any(relative_dark_mask)
        else 0.0
    )
    relative_black_strength = (
        min(
            max(0.0, (background_gray - disk_gray - 8.0) / 24.0),
            float(np.mean(relative_dark_mask[disk_mask])) / 0.42,
            relative_dark_low_saturation / profile.black_low_saturation_fraction_min,
        )
        if relative_black_round
        else 0.0
    )
    relative_white_strength = (
        min(
            max(0.0, (disk_gray - background_gray - 6.0) / 22.0),
            float(np.mean(relative_light_mask[disk_mask])) / 0.42,
        )
        if relative_white_round
        else 0.0
    )

    black_round = _centered_round_mask(dark_mask & disk_mask) or relative_black_round
    white_round = _centered_round_mask(white_mask & disk_mask) or relative_white_round
    is_edge = x in (0, profile.board_size - 1) or y in (0, profile.board_size - 1)
    marker_black_strength = min(
        marker_inner / MARKER_INNER_FRACTION_MIN,
        dark_outer / MARKER_BLACK_OUTER_MIN,
        dark_disk / MARKER_BLACK_DISK_MIN,
        dark_low_saturation / MARKER_BLACK_LOW_SATURATION_MIN,
    )
    marker_black = marker_black_strength >= 1.0
    edge_black_strength = min(
        dark_inner / EDGE_BLACK_INNER_MIN,
        dark_outer / EDGE_BLACK_OUTER_MIN,
        dark_disk / EDGE_BLACK_DISK_MIN,
        dark_low_saturation / profile.black_low_saturation_fraction_min,
    )
    edge_white_strength = min(
        white_inner / EDGE_WHITE_INNER_MIN,
        white_outer / EDGE_WHITE_OUTER_MIN,
        white_disk / EDGE_WHITE_DISK_MIN,
    )
    normal_black_strength = (
        min(
            dark_disk / profile.black_disk_fraction_min,
            dark_low_saturation / profile.black_low_saturation_fraction_min,
        )
        if black_round
        else 0.0
    )
    normal_white_strength = white_disk / profile.white_disk_fraction_min if white_round else 0.0

    if is_edge:
        black_strength = max(edge_black_strength, marker_black_strength, relative_black_strength)
        white_strength = max(edge_white_strength, relative_white_strength)
        black_detected = black_strength >= 1.0 or marker_black
        white_detected = white_strength >= 1.0
    else:
        black_strength = max(normal_black_strength, marker_black_strength, relative_black_strength)
        white_strength = max(normal_white_strength, relative_white_strength)
        black_detected = (black_strength >= 1.0 and black_round) or marker_black
        white_detected = white_strength >= 1.0 and white_round

    black_score = min(1.0, max(0.0, black_strength))
    white_score = min(1.0, max(0.0, white_strength))
    empty_score = max(0.0, 1.0 - max(black_score, white_score))
    if black_detected or white_detected:
        empty_score = min(empty_score, 0.18)
    return CellEvidence(black=black_score, white=white_score, empty=empty_score)


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


def assess_obstruction(
    warped_bgr: np.ndarray, spacing: float, profile: BoardProfile
) -> str | None:
    """Return a reason when a banner or dialog blocks a substantial board area."""

    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    kernel_size = max(5, round(spacing * 0.15))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    dark = (gray <= min(profile.black_gray_max, 60)).astype(np.uint8) * 255
    light = (
        (gray >= max(profile.white_gray_min, 205))
        & (hsv[..., 1] <= profile.white_saturation_max)
    ).astype(np.uint8) * 255
    for mask in (dark, light):
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if _has_large_obstruction(opened, spacing):
            return "board is covered by a popup or banner"
    return None


def _has_large_obstruction(mask: np.ndarray, spacing: float) -> bool:
    component_count, _, statistics, _ = cv2.connectedComponentsWithStats(mask)
    minimum_width = spacing * 3.0
    minimum_height = spacing * 0.55
    minimum_area = spacing**2 * 1.2
    for left, top, width, height, area in statistics[1:component_count]:
        del left, top
        if (
            width >= minimum_width
            and height >= minimum_height
            and area >= minimum_area
        ):
            return True
    return False


def assess_grid(
    warped_bgr: np.ndarray, spacing: float, board_size: int
) -> GridAssessment:
    """Assess expected grid lines using Lab color separation and directional edges."""

    lab = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gradient_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gradient_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    vertical = _directional_grid_score(lab, gradient_x, spacing, board_size, axis=1)
    horizontal = _directional_grid_score(lab, gradient_y, spacing, board_size, axis=0)
    return GridAssessment(vertical_score=vertical, horizontal_score=horizontal)


def _directional_grid_score(
    lab: np.ndarray,
    gradient: np.ndarray,
    spacing: float,
    board_size: int,
    *,
    axis: int,
) -> float:
    """Score one grid direction; axis 1 is vertical, axis 0 is horizontal."""

    scores: list[float] = []
    line_half_width = max(1, round(spacing * 0.025))
    side_offset = max(3, round(spacing * 0.11))
    for index in range(board_size):
        point = round(index * spacing)
        before_start = point - side_offset - line_half_width
        before_end = point - side_offset + line_half_width + 1
        line_start = point - line_half_width
        line_end = point + line_half_width + 1
        after_start = point + side_offset - line_half_width
        after_end = point + side_offset + line_half_width + 1
        if before_start < 0 or after_end > lab.shape[axis]:
            continue

        if axis == 1:
            before_lab = lab[:, before_start:before_end]
            line_lab = lab[:, line_start:line_end]
            after_lab = lab[:, after_start:after_end]
            line_gradient = gradient[:, max(0, line_start - 2) : min(
                gradient.shape[1], line_end + 2
            )]
        else:
            before_lab = lab[before_start:before_end, :]
            line_lab = lab[line_start:line_end, :]
            after_lab = lab[after_start:after_end, :]
            line_gradient = gradient[max(0, line_start - 2) : min(
                gradient.shape[0], line_end + 2
            ), :]

        if not (before_lab.size and line_lab.size and after_lab.size and line_gradient.size):
            continue
        side_color = (before_lab.mean(axis=(0, 1)) + after_lab.mean(axis=(0, 1))) / 2
        line_color = line_lab.mean(axis=(0, 1))
        color_score = min(float(np.linalg.norm(side_color - line_color)) / 18.0, 1.0)
        edge_score = min(float(np.median(line_gradient)) / 16.0, 1.0)
        scores.append(color_score * 0.65 + edge_score * 0.35)
    if not scores:
        return 0.0
    return float(np.median(scores))


class StableStateTracker:
    """Commits stable visual evidence that forms a legal new board state."""

    def __init__(
        self,
        required_frames: int = 3,
        min_confidence: float = 0.70,
        reset_frames: int = 2,
    ) -> None:
        if required_frames < 1:
            raise ValueError("required_frames must be positive")
        self.required_frames = required_frames
        self.min_confidence = min_confidence
        self.reset_frames = reset_frames
        self._samples: list[RecognitionResult] = []
        self._reset_samples: list[BoardState] = []
        self.committed: BoardState | None = None

    def reset(self) -> None:
        self._samples.clear()
        self._reset_samples.clear()
        self.committed = None

    def rebase(self, board: BoardState) -> None:
        """Accept an explicit user correction as the next transition baseline."""

        self._samples.clear()
        self._reset_samples.clear()
        self.committed = board

    def observe(self, result: RecognitionResult) -> tuple[TransitionResult, BoardState | None]:
        if result.obstruction_reason is not None:
            self._samples.clear()
            self._reset_samples.clear()
            return TransitionResult(False, False, result.obstruction_reason), None
        if not result.board_visible:
            self._samples.clear()
            self._reset_samples.clear()
            return TransitionResult(False, False, "board grid is not visible"), None
        if result.confidence < self.min_confidence:
            self._samples.clear()
            return TransitionResult(False, False, "recognition confidence is too low"), None

        self._samples.append(result)
        self._samples = self._samples[-self.required_frames :]
        if len(self._samples) < self.required_frames:
            return TransitionResult(True, False, "waiting for stable frames"), None

        candidate, evidence, stable = self._fuse_samples()
        if not stable:
            return TransitionResult(True, False, "frames are not stable"), None

        transition = validate_transition(self.committed, candidate)
        repaired = False
        if not transition.valid:
            repaired_candidate, repaired_transition = self._unique_legal_repair(candidate, evidence)
            if repaired_candidate is not None and repaired_transition is not None:
                candidate = repaired_candidate
                transition = TransitionResult(
                    True,
                    repaired_transition.changed,
                    f"{repaired_transition.reason}; repaired one ambiguous visual cell",
                    repaired_transition.added_count,
                )
                repaired = True
            elif repaired_transition is not None:
                transition = repaired_transition
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
        if not transition.valid and not repaired:
            return transition, None
        return transition, None

    def _fuse_samples(self) -> tuple[BoardState, tuple[CellEvidence, ...], bool]:
        size = self._samples[-1].board.size
        required_votes = len(self._samples) // 2 + 1
        cells: list[Stone] = []
        evidence: list[CellEvidence] = []
        stable = True
        for index in range(size * size):
            samples = tuple(self._evidence_at(sample, index) for sample in self._samples)
            fused = CellEvidence(
                black=float(np.mean([sample.black for sample in samples])),
                white=float(np.mean([sample.white for sample in samples])),
                empty=float(np.mean([sample.empty for sample in samples])),
            )
            winner = fused.stone
            if sum(sample.stone is winner for sample in samples) < required_votes:
                stable = False
            cells.append(winner)
            evidence.append(fused)
        return BoardState(size=size, cells=tuple(cells)), tuple(evidence), stable

    @staticmethod
    def _evidence_at(result: RecognitionResult, index: int) -> CellEvidence:
        if len(result.cell_evidence) == len(result.board.cells):
            return result.cell_evidence[index]
        confidence = (
            result.cell_confidences[index]
            if len(result.cell_confidences) == len(result.board.cells)
            else result.confidence
        )
        confidence = min(1.0, max(0.0, confidence))
        remainder = (1.0 - confidence) / 2.0
        stone = result.board.cells[index]
        if stone is Stone.BLACK:
            return CellEvidence(confidence, remainder, remainder)
        if stone is Stone.WHITE:
            return CellEvidence(remainder, confidence, remainder)
        return CellEvidence(remainder, remainder, confidence)

    def _unique_legal_repair(
        self, candidate: BoardState, evidence: tuple[CellEvidence, ...]
    ) -> tuple[BoardState | None, TransitionResult | None]:
        alternatives: list[tuple[BoardState, TransitionResult]] = []
        for index, cell in enumerate(evidence):
            if cell.margin > 0.25:
                continue
            runner_up = cell.runner_up()
            runner_up_score = {
                Stone.BLACK: cell.black,
                Stone.WHITE: cell.white,
                Stone.EMPTY: cell.empty,
            }[runner_up]
            if runner_up_score < 0.58:
                continue
            x = index % candidate.size
            y = index // candidate.size
            if runner_up is candidate.at(x, y):
                continue
            alternative = candidate.set_cell(x, y, runner_up)
            transition = validate_transition(self.committed, alternative)
            if transition.valid:
                alternatives.append((alternative, transition))
        if len(alternatives) == 1:
            return alternatives[0]
        if len(alternatives) > 1:
            return None, TransitionResult(False, False, "multiple ambiguous visual repairs")
        return None, None
