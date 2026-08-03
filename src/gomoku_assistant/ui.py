from __future__ import annotations

import concurrent.futures
import sys
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSettings, QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .analysis import (
    AnalysisResult,
    CandidateMove,
    HeuristicAnalyzer,
    ProofStatus,
    RecommendationMode,
)
from .capture import (
    BlankFrameError,
    CaptureError,
    CaptureUnavailableError,
    WindowCaptureSession,
    WindowInfo,
    get_window_info,
    is_blank_frame,
    list_visible_windows,
)
from .domain import BoardCorrectionState, BoardState, CorrectionEvent, Stone, infer_observed_moves
from .engine import (
    MAX_RAPFI_SEARCH_TIME_MS,
    MAX_TOTAL_ANALYSIS_TIME_MS,
    RapfiAnalyzer,
    RapfiConfig,
)
from .profiles import ProfileStore
from .sessions import SessionLogger
from .vision import (
    BoardProfile,
    CellEvidence,
    RecognitionResult,
    StableStateTracker,
    is_valid_board_quadrilateral,
    order_corners,
    recognize_frame,
)


MARKER_COLORS = (QColor("#c73b33"), QColor("#c88810"), QColor("#167e79"))


class InputMode(str, Enum):
    VISION = "vision"
    MANUAL_RELAY = "manual-relay"


@dataclass
class ManualRelayState:
    pending: bool = False
    changing_move: bool = False
    setup: bool = False
    pending_before: BoardState | None = None


def _pixmap_to_bgr(pixmap: QPixmap) -> np.ndarray:
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    array = np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), image.width(), 4)
    return np.ascontiguousarray(array[:, :, [2, 1, 0]])


def _bgr_to_pixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = QImage(
        rgb.data,
        rgb.shape[1],
        rgb.shape[0],
        rgb.strides[0],
        QImage.Format.Format_RGB888,
    )
    return QPixmap.fromImage(image.copy())


class BoardCanvas(QWidget):
    board_edited = Signal(object)
    relay_point_clicked = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(520, 520)
        self._board = BoardState.empty()
        self._candidates: tuple[CandidateMove, ...] = ()
        self._edit_mode: Stone | str = "off"
        self._move_numbers: dict[tuple[int, int], int] = {}
        self._last_move: tuple[int, int] | None = None
        self._correction_points: tuple[tuple[int, int], ...] = ()
        self._ambiguous_points: tuple[tuple[int, int], ...] = ()
        self._danger_points: tuple[tuple[int, int], ...] = ()
        self._pending_point: tuple[int, int] | None = None

    @property
    def board(self) -> BoardState:
        return self._board

    def set_board(self, board: BoardState) -> None:
        self._board = board
        self.update()

    def set_candidates(self, candidates: tuple[CandidateMove, ...]) -> None:
        self._candidates = candidates
        self.update()

    def set_danger_points(self, points: tuple[tuple[int, int], ...]) -> None:
        self._danger_points = points
        self.update()

    def set_pending_point(self, point: tuple[int, int] | None) -> None:
        self._pending_point = point
        self.update()

    def set_edit_mode(self, mode: Stone | str) -> None:
        self._edit_mode = mode

    def set_visual_annotations(
        self,
        correction_points: tuple[tuple[int, int], ...],
        ambiguous_points: tuple[tuple[int, int], ...],
    ) -> None:
        self._correction_points = correction_points
        self._ambiguous_points = ambiguous_points
        self.update()

    def set_move_annotations(
        self, move_numbers: dict[tuple[int, int], int], last_move: tuple[int, int] | None
    ) -> None:
        self._move_numbers = dict(move_numbers)
        self._last_move = last_move
        self.update()

    def _board_rect(self) -> QRectF:
        margin = 34.0
        edge = max(min(self.width(), self.height()) - margin * 2, 1)
        return QRectF((self.width() - edge) / 2, (self.height() - edge) / 2, edge, edge)

    def _point_to_screen(self, x: int, y: int) -> QPointF:
        rect = self._board_rect()
        spacing = rect.width() / (self._board.size - 1)
        return QPointF(rect.left() + x * spacing, rect.top() + y * spacing)

    def _point_from_screen(self, point: QPointF) -> tuple[int, int] | None:
        rect = self._board_rect()
        if not rect.adjusted(-16, -16, 16, 16).contains(point):
            return None
        spacing = rect.width() / (self._board.size - 1)
        x = round((point.x() - rect.left()) / spacing)
        y = round((point.y() - rect.top()) / spacing)
        if self._board.in_bounds(x, y):
            return x, y
        return None

    def mousePressEvent(self, event: Any) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self._point_from_screen(event.position())
        if point is None:
            return
        x, y = point
        if self._edit_mode == "relay":
            self.relay_point_clicked.emit(x, y)
            return
        if self._edit_mode == "off":
            return
        current = self._board.at(x, y)
        if self._edit_mode is Stone.EMPTY:
            next_board = self._board.set_cell(x, y, Stone.EMPTY)
        elif isinstance(self._edit_mode, Stone):
            next_board = self._board.set_cell(x, y, self._edit_mode)
        elif current is Stone.EMPTY:
            try:
                next_board = self._board.place(x, y)
            except ValueError:
                next_board = self._board.set_cell(x, y, Stone.BLACK)
        else:
            next_board = self._board.set_cell(x, y, Stone.EMPTY)
        self.set_board(next_board)
        self.board_edited.emit(next_board)

    def paintEvent(self, _: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f5f5f2"))

        rect = self._board_rect()
        painter.fillRect(rect, QColor("#e4b466"))
        spacing = rect.width() / (self._board.size - 1)
        painter.setPen(QPen(QColor("#3b3024"), max(1.0, spacing * 0.026)))
        for index in range(self._board.size):
            position = rect.left() + index * spacing
            painter.drawLine(QPointF(position, rect.top()), QPointF(position, rect.bottom()))
            position = rect.top() + index * spacing
            painter.drawLine(QPointF(rect.left(), position), QPointF(rect.right(), position))

        star_radius = max(3.0, spacing * 0.08)
        painter.setBrush(QColor("#3b3024"))
        painter.setPen(Qt.PenStyle.NoPen)
        for x, y in ((3, 3), (11, 3), (7, 7), (3, 11), (11, 11)):
            center = self._point_to_screen(x, y)
            painter.drawEllipse(center, star_radius, star_radius)

        stone_radius = spacing * 0.42
        for y in range(self._board.size):
            for x in range(self._board.size):
                stone = self._board.at(x, y)
                if stone is Stone.EMPTY:
                    continue
                center = self._point_to_screen(x, y)
                if stone is Stone.BLACK:
                    painter.setBrush(QColor("#202124"))
                    painter.setPen(QPen(QColor("#111111"), 1.2))
                else:
                    painter.setBrush(QColor("#fbfbf8"))
                    painter.setPen(QPen(QColor("#a8a8a0"), 1.2))
                painter.drawEllipse(center, stone_radius, stone_radius)

        for x, y in self._ambiguous_points:
            if not self._board.in_bounds(x, y):
                continue
            center = self._point_to_screen(x, y)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(
                QPen(QColor("#c88810"), max(1.5, spacing * 0.045), Qt.PenStyle.DashLine)
            )
            painter.drawRect(
                QRectF(
                    center.x() - stone_radius * 0.82,
                    center.y() - stone_radius * 0.82,
                    stone_radius * 1.64,
                    stone_radius * 1.64,
                )
            )

        for x, y in self._correction_points:
            if not self._board.in_bounds(x, y):
                continue
            center = self._point_to_screen(x, y)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#167e79"), max(2.0, spacing * 0.06)))
            painter.drawEllipse(center, stone_radius * 0.93, stone_radius * 0.93)

        for x, y in self._danger_points:
            if not self._board.in_bounds(x, y) or self._board.at(x, y) is not Stone.EMPTY:
                continue
            center = self._point_to_screen(x, y)
            painter.setPen(QPen(QColor("#9d2745"), max(2.0, spacing * 0.065)))
            painter.drawLine(
                QPointF(center.x() - stone_radius * 0.55, center.y() - stone_radius * 0.55),
                QPointF(center.x() + stone_radius * 0.55, center.y() + stone_radius * 0.55),
            )

        if self._pending_point is not None and self._board.in_bounds(*self._pending_point):
            center = self._point_to_screen(*self._pending_point)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#1a6f9f"), max(2.0, spacing * 0.07)))
            painter.drawEllipse(center, stone_radius * 0.9, stone_radius * 0.9)
            painter.drawLine(
                QPointF(center.x() + stone_radius * 0.55, center.y() - stone_radius * 0.55),
                QPointF(center.x() - stone_radius * 0.55, center.y() + stone_radius * 0.55),
            )

        if self._last_move is not None and self._board.in_bounds(*self._last_move):
            center = self._point_to_screen(*self._last_move)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#d4493f"), max(2.0, spacing * 0.055)))
            painter.drawEllipse(center, stone_radius * 0.78, stone_radius * 0.78)

        line = self._board.winning_line()
        if line:
            painter.setPen(QPen(QColor("#267d4b"), max(3.0, spacing * 0.09)))
            painter.drawLine(
                self._point_to_screen(*line.points[0]), self._point_to_screen(*line.points[-1])
            )

        font = painter.font()
        font.setBold(True)
        font.setPixelSize(max(8, round(stone_radius * 1.08)))
        painter.setFont(font)
        for (x, y), number in self._move_numbers.items():
            if self._board.at(x, y) is Stone.EMPTY:
                continue
            center = self._point_to_screen(x, y)
            color = QColor("#f5d66f") if self._board.at(x, y) is Stone.BLACK else QColor("#5a4630")
            painter.setPen(QPen(color))
            painter.drawText(
                QRectF(
                    center.x() - stone_radius,
                    center.y() - stone_radius,
                    stone_radius * 2,
                    stone_radius * 2,
                ),
                Qt.AlignmentFlag.AlignCenter,
                str(number),
            )

        for candidate in self._candidates:
            if not self._board.in_bounds(candidate.x, candidate.y):
                continue
            center = self._point_to_screen(candidate.x, candidate.y)
            color = MARKER_COLORS[min(candidate.rank - 1, len(MARKER_COLORS) - 1)]
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(color, max(2.0, spacing * 0.08)))
            painter.drawEllipse(center, stone_radius * 0.72, stone_radius * 0.72)
            painter.setPen(QPen(color))
            painter.drawText(
                QRectF(
                    center.x() - stone_radius,
                    center.y() - stone_radius,
                    stone_radius * 2,
                    stone_radius * 2,
                ),
                Qt.AlignmentFlag.AlignCenter,
                str(candidate.rank),
            )


class CalibrationCanvas(QWidget):
    points_changed = Signal(int)

    def __init__(self, frame: np.ndarray, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(720, 540)
        self._pixmap = _bgr_to_pixmap(frame)
        self._frame = frame
        self._points: list[tuple[float, float]] = []
        self._display_rect = QRectF()

    @property
    def points(self) -> tuple[tuple[float, float], ...]:
        return tuple(self._points)

    def clear_points(self) -> None:
        self._points.clear()
        self.points_changed.emit(0)
        self.update()

    def mousePressEvent(self, event: Any) -> None:
        if event.button() != Qt.MouseButton.LeftButton or len(self._points) >= 4:
            return
        if not self._display_rect.contains(event.position()):
            return
        point = event.position()
        normalized_x = (point.x() - self._display_rect.left()) / self._display_rect.width()
        normalized_y = (point.y() - self._display_rect.top()) / self._display_rect.height()
        self._points.append(
            (normalized_x * self._pixmap.width(), normalized_y * self._pixmap.height())
        )
        self.points_changed.emit(len(self._points))
        self.update()

    def paintEvent(self, _: Any) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#202124"))
        source_size = self._pixmap.size()
        target = source_size.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self._display_rect = QRectF(
            (self.width() - target.width()) / 2,
            (self.height() - target.height()) / 2,
            target.width(),
            target.height(),
        )
        painter.drawPixmap(self._display_rect.toRect(), self._pixmap)
        if len(self._points) == 4:
            source = np.array(order_corners(self._points), dtype=np.float32)
            edge = 840.0
            destination = np.array(
                [(0, 0), (edge, 0), (edge, edge), (0, edge)],
                dtype=np.float32,
            )
            inverse = cv2.getPerspectiveTransform(destination, source)
            grid = np.array(
                [
                    [[x * edge / 14, y * edge / 14]]
                    for y in range(15)
                    for x in range(15)
                ],
                dtype=np.float32,
            )
            projected = cv2.perspectiveTransform(grid, inverse).reshape(-1, 2)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#16a6a0"), 1.2))
            for x, y in projected:
                point = QPointF(
                    self._display_rect.left() + x / self._pixmap.width() * self._display_rect.width(),
                    self._display_rect.top() + y / self._pixmap.height() * self._display_rect.height(),
                )
                painter.drawEllipse(point, 2.2, 2.2)
        for index, (x, y) in enumerate(self._points, start=1):
            point = QPointF(
                self._display_rect.left() + x / self._pixmap.width() * self._display_rect.width(),
                self._display_rect.top() + y / self._pixmap.height() * self._display_rect.height(),
            )
            painter.setBrush(QColor("#d6473d"))
            painter.setPen(QPen(QColor("#ffffff"), 1.5))
            painter.drawEllipse(point, 8, 8)
            painter.drawText(QRectF(point.x() + 10, point.y() - 12, 30, 24), str(index))


class CalibrationDialog(QDialog):
    def __init__(self, frame: np.ndarray, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibrate 15x15 board")
        self._frame = frame
        self._canvas = CalibrationCanvas(frame, self)
        self._hint = QLabel("Click top-left, top-right, bottom-right, then bottom-left intersections.")
        self._count = QLabel("0 / 4")
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save
        )
        self._save_button = self._buttons.button(QDialogButtonBox.StandardButton.Save)
        self._save_button.setEnabled(False)
        clear_button = QPushButton("Clear points")
        clear_button.clicked.connect(self._canvas.clear_points)
        self._buttons.rejected.connect(self.reject)
        self._buttons.accepted.connect(self.accept)
        self._canvas.points_changed.connect(self._on_points_changed)

        layout = QVBoxLayout(self)
        layout.addWidget(self._hint)
        layout.addWidget(self._canvas, 1)
        footer = QHBoxLayout()
        footer.addWidget(self._count)
        footer.addStretch(1)
        footer.addWidget(clear_button)
        layout.addLayout(footer)
        layout.addWidget(self._buttons)
        self.resize(920, 760)

    def _on_points_changed(self, count: int) -> None:
        self._count.setText(f"{count} / 4")
        if count == 4:
            if not is_valid_board_quadrilateral(self._canvas.points, self._frame.shape):
                self._save_button.setEnabled(False)
                self._hint.setText(
                    "The four points do not form a valid board. Clear them and click the outer intersections."
                )
                return

            score = self.profile().grid_score_baseline or 0.0
            self._save_button.setEnabled(True)
            if score >= 0.35:
                self._hint.setText(
                    f"Good grid signal {score:.0%}. Cyan points are on intersections; save calibration."
                )
            elif score >= 0.08:
                self._hint.setText(
                    f"Low-contrast grid signal {score:.0%}. If the cyan points align, save calibration."
                )
            else:
                self._hint.setText(
                    f"Very low grid signal {score:.0%}. The points may still be correct; save to test live sync."
                )
        else:
            self._save_button.setEnabled(False)
            self._hint.setText(
                "Click top-left, top-right, bottom-right, then bottom-left intersections."
            )

    def profile(self) -> BoardProfile:
        if len(self._canvas.points) != 4:
            raise ValueError("Four calibration points are required.")
        profile = BoardProfile(board_size=15, corners=order_corners(self._canvas.points))
        baseline = recognize_frame(self._frame, profile).grid_score
        return replace(profile, grid_score_baseline=baseline)


class MainWindow(QMainWindow):
    analysis_ready = Signal(int, object)
    analysis_finished = Signal()
    engine_warmed = Signal(int, object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Gomoku Training Assistant")
        self.resize(1220, 820)
        self._board = BoardState.empty()
        self._candidates: tuple[CandidateMove, ...] = ()
        self._move_numbers: dict[tuple[int, int], int] = {}
        self._last_move: tuple[int, int] | None = None
        self._corrections = BoardCorrectionState()
        self._last_raw_recognition: RecognitionResult | None = None
        self._raw_reset_samples: list[BoardState] = []
        self._profile_store = ProfileStore(Path.cwd() / "profiles")
        self._profile: BoardProfile | None = None
        self._last_frame: np.ndarray | None = None
        self._last_window: WindowInfo | None = None
        self._capture_session = WindowCaptureSession()
        self._tracker = StableStateTracker()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._analysis_version = 0
        self._pending_analyses = 0
        self._warmup_version = 0
        self._observation_waiting_for_engine = False
        self._input_mode = InputMode.VISION
        self._manual = ManualRelayState()
        self._manual_history: list[BoardState] = []
        self._session = SessionLogger(Path.cwd() / "sessions")
        self._settings = QSettings("Nocoding0", "GomokuTrainingAssistant")
        self._rapfi_path = self._default_rapfi_path()
        self._rapfi_analyzer: RapfiAnalyzer | None = None

        self._board_canvas = BoardCanvas()
        self._board_canvas.board_edited.connect(self._on_manual_board_edit)
        self._board_canvas.relay_point_clicked.connect(self._on_manual_relay_point)
        self._preview = QLabel("No frame captured")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(330, 230)
        self._preview.setFrameShape(QFrame.Shape.StyledPanel)
        self._preview.setStyleSheet("background: #202124; color: #dddddd;")

        self._window_combo = QComboBox()
        self._window_combo.currentIndexChanged.connect(self._on_target_window_changed)
        self._refresh_windows_button = QPushButton("Refresh windows")
        self._refresh_windows_button.clicked.connect(self.refresh_windows)
        self._capture_button = QPushButton("Capture frame")
        self._capture_button.clicked.connect(self.capture_frame)
        self._load_image_button = QPushButton("Open screenshot")
        self._load_image_button.clicked.connect(self.open_screenshot)
        self._calibrate_button = QPushButton("Calibrate")
        self._calibrate_button.clicked.connect(self.calibrate)
        self._observe_button = QPushButton("Start observing")
        self._observe_button.setCheckable(True)
        self._observe_button.toggled.connect(self._toggle_observation)
        self._analyze_button = QPushButton("Analyze position")
        self._analyze_button.clicked.connect(self.analyze_current_board)
        self._clear_button = QPushButton("Clear board")
        self._clear_button.clicked.connect(self.clear_board)
        self._engine_button = QPushButton("Select Rapfi.exe")
        self._engine_button.clicked.connect(self.select_rapfi)

        self._black_search_time = QSpinBox()
        self._black_search_time.setRange(1, MAX_RAPFI_SEARCH_TIME_MS // 1000)
        self._black_search_time.setSuffix(" s")
        self._black_search_time.setValue(self._saved_int("rapfi_black_seconds", 8, 1, 15))
        self._white_search_time = QSpinBox()
        self._white_search_time.setRange(1, MAX_RAPFI_SEARCH_TIME_MS // 1000)
        self._white_search_time.setSuffix(" s")
        self._white_search_time.setValue(self._saved_int("rapfi_white_seconds", 15, 1, 15))
        self._rapfi_threads = QSpinBox()
        self._rapfi_threads.setRange(1, 16)
        self._rapfi_threads.setValue(self._saved_int("rapfi_threads", 8, 1, 16))
        self._rapfi_hash = QComboBox()
        for megabytes in (128, 256, 512, 1024):
            self._rapfi_hash.addItem(f"{megabytes} MB", megabytes)
        saved_hash = self._saved_int("rapfi_hash_mb", 512, 128, 1024)
        hash_index = self._rapfi_hash.findData(saved_hash)
        self._rapfi_hash.setCurrentIndex(hash_index if hash_index >= 0 else 2)
        self._black_search_time.valueChanged.connect(self._save_engine_settings)
        self._white_search_time.valueChanged.connect(self._save_engine_settings)
        self._rapfi_threads.valueChanged.connect(self._save_engine_settings)
        self._rapfi_hash.currentIndexChanged.connect(self._save_engine_settings)
        self._edit_mode = QComboBox()
        self._edit_mode.addItem("Off", "off")
        self._edit_mode.addItem("Auto move", "auto")
        self._edit_mode.addItem("Place black", Stone.BLACK)
        self._edit_mode.addItem("Place white", Stone.WHITE)
        self._edit_mode.addItem("Erase", Stone.EMPTY)
        self._edit_mode.currentIndexChanged.connect(self._update_edit_mode)
        self._manual_setup_tool = QComboBox()
        self._manual_setup_tool.addItem("放置黑棋", Stone.BLACK)
        self._manual_setup_tool.addItem("放置白棋", Stone.WHITE)
        self._manual_setup_tool.addItem("擦除", Stone.EMPTY)
        self._manual_setup_tool.currentIndexChanged.connect(self._update_edit_mode)
        self._undo_correction_button = QPushButton("Undo correction")
        self._undo_correction_button.clicked.connect(self.undo_correction)
        self._clear_corrections_button = QPushButton("Clear corrections")
        self._clear_corrections_button.clicked.connect(self.clear_corrections)
        self._my_color = QComboBox()
        self._my_color.addItem("Choose my color", None)
        self._my_color.addItem("Black", Stone.BLACK)
        self._my_color.addItem("White", Stone.WHITE)
        self._my_color.addItem("Analyze both colors", "both")
        self._my_color.currentIndexChanged.connect(self._on_my_color_changed)
        self._manual_start_button = QPushButton("新建手动对局")
        self._manual_start_button.clicked.connect(self.start_manual_game)
        self._manual_continue_button = QPushButton("从当前局面继续")
        self._manual_continue_button.clicked.connect(self.continue_manual_game)
        self._manual_undo_button = QPushButton("撤销上一手")
        self._manual_undo_button.clicked.connect(self.undo_manual_move)
        self._manual_change_button = QPushButton("改走本手")
        self._manual_change_button.clicked.connect(self.change_manual_move)
        self._manual_setup_button = QPushButton("编辑局面")
        self._manual_setup_button.setCheckable(True)
        self._manual_setup_button.toggled.connect(self._toggle_manual_setup)
        self._manual_finish_setup_button = QPushButton("完成局面编辑")
        self._manual_finish_setup_button.clicked.connect(self.finish_manual_setup)

        self._profile_status = QLabel(self._profile_status_text())
        self._engine_status = QLabel(self._engine_status_text())
        self._status = QLabel("Ready")
        self._status.setWordWrap(True)
        self._status.setMinimumHeight(42)
        self._status.setStyleSheet(
            "background: #edf3f2; border: 1px solid #b7c7c3; padding: 5px;"
        )
        self._candidate_text = QLabel("No analysis yet")
        self._candidate_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._candidate_text.setWordWrap(True)

        control_group = QGroupBox("自动识别")
        controls = QFormLayout(control_group)
        controls.addRow("目标窗口", self._window_combo)
        controls.addRow("", self._refresh_windows_button)
        controls.addRow("", self._capture_button)
        controls.addRow("", self._load_image_button)
        controls.addRow("", self._calibrate_button)
        controls.addRow("标定状态", self._profile_status)
        controls.addRow("", self._observe_button)
        controls.addRow("识别修正", self._edit_mode)
        controls.addRow("", self._undo_correction_button)
        controls.addRow("", self._clear_corrections_button)
        controls.addRow("我的棋色", self._my_color)
        controls.addRow("", self._analyze_button)
        controls.addRow("", self._clear_button)

        manual_group = QGroupBox("手动搬运")
        manual_layout = QVBoxLayout(manual_group)
        manual_layout.addWidget(QLabel("把真实棋盘的对方落子点到左侧棋盘。助手会自动采用第 1 推荐。"))
        manual_layout.addWidget(self._manual_start_button)
        manual_layout.addWidget(self._manual_continue_button)
        manual_layout.addWidget(self._manual_undo_button)
        manual_layout.addWidget(self._manual_change_button)
        manual_layout.addWidget(self._manual_setup_button)
        manual_layout.addWidget(self._manual_setup_tool)
        manual_layout.addWidget(self._manual_finish_setup_button)
        manual_layout.addStretch(1)

        advanced = QGroupBox("高级引擎设置")
        advanced.setCheckable(True)
        advanced.setChecked(False)
        advanced_layout = QFormLayout(advanced)
        advanced_layout.addRow("黑棋思考", self._black_search_time)
        advanced_layout.addRow("白棋思考", self._white_search_time)
        advanced_layout.addRow("Rapfi 线程", self._rapfi_threads)
        advanced_layout.addRow("Rapfi 哈希", self._rapfi_hash)
        advanced_layout.addRow("", self._engine_button)
        advanced_layout.addRow("引擎", self._engine_status)
        advanced.toggled.connect(lambda checked: [child.setVisible(checked) for child in advanced.findChildren(QWidget) if child is not advanced])
        for child in advanced.findChildren(QWidget):
            child.setVisible(False)

        suggestion_group = QGroupBox("推荐落点")
        suggestion_layout = QVBoxLayout(suggestion_group)
        suggestion_layout.addWidget(self._candidate_text)
        suggestion_layout.addStretch(1)

        tabs = QTabWidget()
        vision_tab = QWidget()
        vision_layout = QVBoxLayout(vision_tab)
        vision_layout.addWidget(control_group)
        vision_layout.addWidget(QLabel("最新来源画面"))
        vision_layout.addWidget(self._preview, 1)
        manual_tab = QWidget()
        manual_tab_layout = QVBoxLayout(manual_tab)
        manual_tab_layout.addWidget(manual_group)
        tabs.addTab(vision_tab, "自动识别")
        tabs.addTab(manual_tab, "手动搬运")
        tabs.currentChanged.connect(self._on_input_mode_changed)
        self._mode_tabs = tabs

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.addWidget(suggestion_group)
        right_layout.addWidget(self._status)
        right_layout.addWidget(tabs, 1)
        right_layout.addWidget(advanced)

        splitter = QSplitter()
        splitter.addWidget(self._board_canvas)
        splitter.addWidget(right_column)
        splitter.setSizes([760, 420])
        self.setCentralWidget(splitter)

        self._capture_timer = QTimer(self)
        self._capture_timer.setInterval(250)
        self._capture_timer.timeout.connect(self._observe_tick)
        self.analysis_ready.connect(self._on_analysis_ready)
        self.analysis_finished.connect(self._on_analysis_finished)
        self.engine_warmed.connect(self._on_engine_warmed)
        self._refresh_shortcut = QShortcut(QKeySequence("F5"), self)
        self._refresh_shortcut.activated.connect(self.refresh_windows)
        self.refresh_windows()
        self._update_edit_mode()
        self._refresh_manual_controls()
        self._refresh_visual_annotations()

    def _default_rapfi_path(self) -> Path | None:
        candidate = Path.cwd() / "vendor" / "rapfi" / "Rapfi.exe"
        return candidate if candidate.is_file() else None

    def _saved_int(self, key: str, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(self._settings.value(key, default))
        except (TypeError, ValueError):
            value = default
        return min(max(value, minimum), maximum)

    def _save_engine_settings(self, _: int = 0) -> None:
        self._settings.setValue("rapfi_black_seconds", self._black_search_time.value())
        self._settings.setValue("rapfi_white_seconds", self._white_search_time.value())
        self._settings.setValue("rapfi_threads", self._rapfi_threads.value())
        self._settings.setValue("rapfi_hash_mb", int(self._rapfi_hash.currentData()))

    def _current_rapfi_config(self) -> RapfiConfig | None:
        if self._rapfi_path is None or not self._rapfi_path.is_file():
            return None
        return RapfiConfig(
            executable=self._rapfi_path,
            time_ms=MAX_RAPFI_SEARCH_TIME_MS,
            threads=self._rapfi_threads.value(),
            hash_kib=int(self._rapfi_hash.currentData()) * 1024,
        )

    def _search_budget_ms(self, board: BoardState) -> int:
        seconds = (
            self._black_search_time.value()
            if board.side_to_move() is Stone.BLACK
            else self._white_search_time.value()
        )
        return min(seconds * 1000, MAX_RAPFI_SEARCH_TIME_MS)

    def _set_engine_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self._black_search_time,
            self._white_search_time,
            self._rapfi_threads,
            self._rapfi_hash,
        ):
            widget.setEnabled(enabled)

    def _reset_move_history(self, start_new_game: bool = False) -> None:
        self._move_numbers.clear()
        self._last_move = None
        self._board_canvas.set_move_annotations({}, None)
        if start_new_game:
            color = self._my_color.currentData()
            self._session.start_game(color if isinstance(color, Stone) else None)

    def _record_observed_moves(
        self, previous: BoardState | None, current: BoardState, source: str
    ) -> None:
        moves = infer_observed_moves(previous, current)
        if not moves:
            return
        for move in moves:
            if move.certain and move.number is not None:
                self._move_numbers[(move.x, move.y)] = move.number
        numbered = [move for move in moves if move.certain and move.number is not None]
        latest = max(numbered, key=lambda move: move.number) if numbered else moves[-1]
        self._last_move = (latest.x, latest.y)
        self._board_canvas.set_move_annotations(self._move_numbers, self._last_move)
        self._session.record_moves(moves, source=source)

    def _raw_board(self) -> BoardState:
        return self._last_raw_recognition.board if self._last_raw_recognition else self._board

    def _apply_corrections(self, recognition: RecognitionResult) -> RecognitionResult:
        if self._corrections.count == 0:
            return recognition
        board = self._corrections.apply(recognition.board)
        confidences = list(recognition.cell_confidences)
        evidence = list(recognition.cell_evidence)
        for (x, y), stone in self._corrections.overrides.items():
            index = board.index(x, y)
            if len(confidences) == len(board.cells):
                confidences[index] = 1.0
            if len(evidence) == len(board.cells):
                evidence[index] = CellEvidence(
                    black=1.0 if stone is Stone.BLACK else 0.0,
                    white=1.0 if stone is Stone.WHITE else 0.0,
                    empty=1.0 if stone is Stone.EMPTY else 0.0,
                )
        correction_points = set(self._corrections.points)
        return replace(
            recognition,
            board=board,
            confidence=float(np.mean(confidences)) if confidences else recognition.confidence,
            cell_confidences=tuple(confidences),
            cell_evidence=tuple(evidence),
            ambiguous_points=tuple(
                point for point in recognition.ambiguous_points if point not in correction_points
            ),
        )

    def _refresh_visual_annotations(self, recognition: RecognitionResult | None = None) -> None:
        current = recognition or self._last_raw_recognition
        ambiguous = current.ambiguous_points if current is not None else ()
        correction_points = set(self._corrections.points)
        self._board_canvas.set_visual_annotations(
            self._corrections.points,
            tuple(point for point in ambiguous if point not in correction_points),
        )
        self._undo_correction_button.setEnabled(self._corrections.can_undo)
        self._clear_corrections_button.setEnabled(self._corrections.count > 0)

    def _record_correction(self, event: CorrectionEvent) -> None:
        self._session.record_correction(event)

    def _record_manual_change(self, previous: BoardState, current: BoardState) -> None:
        moves = infer_observed_moves(previous, current)
        if moves:
            self._record_observed_moves(previous, current, source="manual_correction")
            return
        changed = [
            (x, y)
            for y in range(current.size)
            for x in range(current.size)
            if previous.at(x, y) is not current.at(x, y)
        ]
        for point in changed:
            self._move_numbers.pop(point, None)
        if self._last_move in changed:
            numbered = [
                (number, point) for point, number in self._move_numbers.items() if current.at(*point) is not Stone.EMPTY
            ]
            self._last_move = max(numbered)[1] if numbered else None
        self._board_canvas.set_move_annotations(self._move_numbers, self._last_move)

    def _set_corrected_board(self, board: BoardState, status: str) -> None:
        self._tracker.rebase(board)
        self._set_board(board, analyze=False)
        self._refresh_visual_annotations()
        if board.is_terminal():
            self._session.finish_game(board)
            self._status.setText(status)
            return
        if not board.is_count_legal():
            self._candidate_text.setText("Manual correction is incomplete: fix black/white counts before analysis.")
            self._status.setText(status)
            return
        self._status.setText(status)
        if self._should_analyze_turn(board):
            self.analyze_current_board()
        else:
            next_side = "Black" if board.side_to_move() is Stone.BLACK else "White"
            self._candidate_text.setText(f"Manual correction saved. Waiting for {next_side}.")

    def _clear_correction_state(self, *, record: bool) -> tuple[CorrectionEvent, ...]:
        events = self._corrections.clear()
        if record:
            for event in events:
                self._record_correction(event)
        self._refresh_visual_annotations()
        return events

    def _maybe_start_new_game_from_raw_frame(self, recognition: RecognitionResult) -> bool:
        if self._corrections.count == 0:
            self._raw_reset_samples.clear()
            return False
        raw = recognition.board
        if (
            recognition.obstruction_reason is not None
            or not recognition.board_visible
            or recognition.confidence < self._tracker.min_confidence
            or not raw.is_count_legal()
            or sum(raw.counts()) >= sum(self._board.counts())
        ):
            self._raw_reset_samples.clear()
            return False
        self._raw_reset_samples.append(raw)
        self._raw_reset_samples = self._raw_reset_samples[-self._tracker.reset_frames :]
        if len(self._raw_reset_samples) != self._tracker.reset_frames or len(
            set(sample.cells for sample in self._raw_reset_samples)
        ) != 1:
            return False
        self._clear_correction_state(record=False)
        self._tracker.reset()
        self._reset_move_history(start_new_game=True)
        self._set_board(BoardState.empty(), analyze=False)
        self._raw_reset_samples.clear()
        self._status.setText("New game detected. Cleared previous manual corrections.")
        return True

    def _save_profile(self) -> None:
        if self._profile is None:
            return
        if self._profile.window_title is None:
            self._status.setText(
                "Screenshot calibration is active only for this session. Select a window and calibrate to save it."
            )
            return
        self._profile_store.save(self._profile)

    def _profile_status_text(self) -> str:
        if self._profile is None:
            return "Not calibrated"
        if self._profile.schema_version < 3 or self._profile.grid_score_baseline is None:
            return "Legacy profile: recalibrate"
        if self._profile.source_width is None:
            return "Legacy profile: recalibrate"
        return f"15x15 calibrated: {self._profile.window_title}"

    def _engine_status_text(self) -> str:
        if self._rapfi_path and self._rapfi_path.is_file():
            return self._rapfi_path.name
        return "Local tactical heuristic"

    def _selected_handle(self) -> int | None:
        data = self._window_combo.currentData()
        return int(data) if data is not None else None

    def refresh_windows(self) -> None:
        current = self._selected_handle()
        blocker = QSignalBlocker(self._window_combo)
        self._window_combo.clear()
        own_handle = int(self.winId())
        for window in list_visible_windows(exclude_handle=own_handle):
            self._window_combo.addItem(
                f"{window.title} [{window.width} x {window.height}]",
                window.handle,
            )
        if current is not None:
            index = self._window_combo.findData(current)
            if index >= 0:
                self._window_combo.setCurrentIndex(index)
        del blocker
        selected = self._selected_handle()
        if selected != current:
            self._on_target_window_changed(self._window_combo.currentIndex())
        if self._window_combo.count() == 0:
            self._status.setText("No eligible Windows application window found.")

    def _on_target_window_changed(self, _: int) -> None:
        self._capture_session.stop()
        self._last_frame = None
        self._last_window = None
        self._last_raw_recognition = None
        self._raw_reset_samples.clear()
        self._clear_correction_state(record=False)
        self._tracker.reset()
        self._reset_move_history()
        self._set_board(BoardState.empty(), analyze=False)
        self._candidate_text.setText("No analysis yet")
        self._preview.setPixmap(QPixmap())
        self._preview.setText("No frame captured")
        handle = self._selected_handle()
        window = get_window_info(handle) if handle is not None else None
        self._profile = self._profile_store.load_for(window) if window else None
        self._profile_status.setText(self._profile_status_text())
        if window:
            self._status.setText(
                "Selected target. Capture a frame, then calibrate if no profile was loaded."
            )

    def capture_frame(self) -> np.ndarray | None:
        handle = self._selected_handle()
        if handle is None:
            self._status.setText("Select a target window first.")
            return None
        window = get_window_info(handle)
        if window is None or window.width <= 0 or window.height <= 0:
            self._status.setText("Target window is no longer available.")
            return None

        try:
            captured = self._capture_session.latest_frame(window)
            frame = captured.frame_bgr
        except CaptureUnavailableError:
            frame = self._capture_with_qt_fallback(handle)
            if frame is None:
                return None
        except (BlankFrameError, CaptureError) as error:
            self._last_frame = None
            self._last_window = None
            self._preview.setPixmap(QPixmap())
            self._preview.setText("No valid frame captured")
            self._status.setText(str(error))
            return None

        self._last_frame = frame
        self._last_window = window
        if self._profile is None:
            self._profile = self._profile_store.load_for(window)
            self._profile_status.setText(self._profile_status_text())
        self._preview.setPixmap(
            _bgr_to_pixmap(frame).scaled(
                self._preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        return self._last_frame

    def _capture_with_qt_fallback(self, handle: int) -> np.ndarray | None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self._status.setText("No primary screen is available.")
            return None
        pixmap = screen.grabWindow(handle)
        if pixmap.isNull():
            self._status.setText("Windows did not return a capturable frame for this window.")
            return None
        frame = _pixmap_to_bgr(pixmap)
        if is_blank_frame(frame):
            self._status.setText(
                "Qt fallback captured a blank frame. Install windows-capture and retry."
            )
            return None
        return frame

    def open_screenshot(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open screenshot",
            str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not filename:
            return
        data = np.fromfile(filename, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            QMessageBox.warning(self, "Cannot open image", "OpenCV could not decode this image.")
            return
        self._last_frame = frame
        self._last_window = None
        self._preview.setPixmap(
            _bgr_to_pixmap(frame).scaled(
                self._preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._status.setText(f"Loaded screenshot: {Path(filename).name}")

    def calibrate(self) -> None:
        if self._last_frame is None:
            self.capture_frame()
        if self._last_frame is None:
            return
        dialog = CalibrationDialog(self._last_frame, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._profile = replace(
            dialog.profile(),
            source_width=self._last_frame.shape[1],
            source_height=self._last_frame.shape[0],
            window_title=self._last_window.title if self._last_window else None,
        )
        self._tracker.reset()
        self._clear_correction_state(record=False)
        self._invalidate_analysis()
        self._save_profile()
        self._profile_status.setText(self._profile_status_text())
        self._status.setText("Calibration saved. Start observation or capture a new frame.")

    def _toggle_observation(self, enabled: bool) -> None:
        if enabled and self._profile is None:
            self._observe_button.setChecked(False)
            QMessageBox.information(self, "Calibration required", "Calibrate the 15x15 board first.")
            return
        if (
            enabled
            and (
                self._profile.schema_version < 3
                or self._profile.grid_score_baseline is None
            )
        ):
            self._observe_button.setChecked(False)
            QMessageBox.information(
                self,
                "Recalibration required",
                "This saved profile predates adaptive grid validation. Capture a frame and calibrate again.",
            )
            return
        if enabled and self._my_color.currentData() is None:
            self._observe_button.setChecked(False)
            QMessageBox.information(
                self,
                "Choose my color",
                "Choose Black, White, or Analyze both colors before observation starts.",
            )
            return
        if enabled:
            frame = self.capture_frame()
            if frame is None or self._last_window is None:
                self._observe_button.setChecked(False)
                return
            if not self._profile.matches_source(frame, self._last_window.title):
                self._observe_button.setChecked(False)
                QMessageBox.information(
                    self,
                    "Calibration required",
                    "The selected window or capture size changed. Capture a frame and calibrate again.",
                )
                return
        if enabled:
            self._clear_correction_state(record=False)
            self._last_raw_recognition = None
            self._raw_reset_samples.clear()
            self._tracker.reset()
            self._set_board(BoardState.empty(), analyze=False)
            self._reset_move_history(start_new_game=True)
            config = self._current_rapfi_config()
            if config is None:
                self._begin_observation()
                return
            if self._rapfi_analyzer is None or self._rapfi_analyzer.config != config:
                if self._rapfi_analyzer is not None:
                    self._rapfi_analyzer.close(wait_timeout_s=0)
                self._rapfi_analyzer = RapfiAnalyzer(config)
            self._observation_waiting_for_engine = True
            self._warmup_version += 1
            version = self._warmup_version
            self._observe_button.setText("Preparing engine...")
            self._clear_button.setText("Resync board")
            self._candidate_text.setText("Preparing Rapfi before observation starts.")
            self._status.setText("Starting Rapfi before the move clock begins.")
            self._set_engine_controls_enabled(False)
            future = self._executor.submit(self._rapfi_analyzer.warm_up)

            def warmed(task: concurrent.futures.Future[None]) -> None:
                try:
                    payload: object = task.result()
                except Exception as error:
                    payload = error
                self.engine_warmed.emit(version, payload)

            future.add_done_callback(warmed)
        else:
            self._warmup_version += 1
            self._observation_waiting_for_engine = False
            self._capture_timer.stop()
            self._invalidate_analysis()
            self._observe_button.setText("Start observing")
            self._clear_button.setText("Clear board")
            if self._pending_analyses == 0:
                self._set_engine_controls_enabled(True)
            self._status.setText("Observation paused.")

    def _begin_observation(self) -> None:
        self._observation_waiting_for_engine = False
        self._observe_button.setText("Stop observing")
        self._clear_button.setText("Resync board")
        self._candidate_text.setText("Waiting for a stable board frame.")
        self._capture_timer.start()
        self._set_engine_controls_enabled(True)
        self._status.setText("Observing target window.")

    def _on_engine_warmed(self, version: int, payload: object) -> None:
        if version != self._warmup_version or not self._observe_button.isChecked():
            return
        self._set_engine_controls_enabled(True)
        if isinstance(payload, Exception):
            self._observation_waiting_for_engine = False
            self._observe_button.setChecked(False)
            self._candidate_text.setText(f"Engine warm-up failed: {payload}")
            self._status.setText("Rapfi was not ready; observation did not start.")
            return
        self._begin_observation()

    def _observe_tick(self) -> None:
        if self._profile is None:
            return
        frame = self.capture_frame()
        if frame is None:
            return
        try:
            raw_recognition = recognize_frame(frame, self._profile)
            self._last_raw_recognition = raw_recognition
            self._maybe_start_new_game_from_raw_frame(raw_recognition)
            recognition = self._apply_corrections(raw_recognition)
            self._refresh_visual_annotations(raw_recognition)
            transition, committed = self._tracker.observe(recognition)
        except Exception as error:
            self._status.setText(f"Recognition paused: {error}")
            return
        if committed is not None:
            previous = self._board
            if transition.reason == "new game detected":
                self._clear_correction_state(record=False)
                self._reset_move_history(start_new_game=True)
                previous = BoardState.empty()
            self._record_observed_moves(previous, committed, source="vision")
            is_terminal = committed.is_terminal()
            should_analyze = not is_terminal and self._should_analyze_turn(committed)
            self._set_board(committed, analyze=should_analyze)
            black, white = committed.counts()
            if is_terminal:
                self._session.finish_game(committed)
                winner = committed.winner()
                self._candidate_text.setText(
                    "Game over: "
                    + ("Black wins." if winner is Stone.BLACK else "White wins." if winner else "Draw.")
                )
                self._status.setText(f"Synced B{black}/W{white}; game finished.")
                return
            next_side = "Black" if committed.side_to_move() is Stone.BLACK else "White"
            if should_analyze:
                self._status.setText(
                    f"Synced B{black}/W{white}; {transition.reason}; analyzing {next_side}."
                )
            else:
                self._candidate_text.setText(f"Synced. Waiting for opponent ({next_side}).")
                self._status.setText(
                    f"Synced B{black}/W{white}; {transition.reason}; opponent {next_side} to move."
                )
        elif not transition.valid:
            self._invalidate_analysis()
            if recognition.obstruction_reason is not None:
                black, white = self._board.counts()
                self._status.setText("Paused: board is covered; waiting for a clear frame.")
                self._candidate_text.setText(
                    f"Last confirmed board: B{black}/W{white}. Waiting for the popup to clear."
                )
            else:
                black, white = raw_recognition.board.counts()
                confirmed_black, confirmed_white = self._board.counts()
                ambiguous = ", ".join(
                    self._board.coordinate_name(x, y)
                    for x, y in raw_recognition.ambiguous_points[:4]
                )
                ambiguity_text = f"; check {ambiguous}" if ambiguous else ""
                self._status.setText(
                    f"Not synced: {transition.reason}; grid {recognition.grid_score:.0%}; "
                    f"confidence {recognition.confidence:.0%}; detected B{black}/W{white}; "
                    f"last confirmed B{confirmed_black}/W{confirmed_white}{ambiguity_text}."
                )
        else:
            black, white = raw_recognition.board.counts()
            ambiguous = ", ".join(
                self._board.coordinate_name(x, y) for x, y in raw_recognition.ambiguous_points[:4]
            )
            ambiguity_text = f"; check {ambiguous}" if ambiguous else ""
            self._status.setText(
                f"Watching: grid {recognition.grid_score:.0%}; confidence "
                f"{recognition.confidence:.0%}; detected B{black}/W{white}; {transition.reason}{ambiguity_text}."
            )

    def _should_analyze_turn(self, board: BoardState) -> bool:
        my_color = self._my_color.currentData()
        return my_color == "both" or board.side_to_move() is my_color

    def _on_my_color_changed(self, _: int) -> None:
        self._invalidate_analysis()
        if self._input_mode is InputMode.MANUAL_RELAY:
            self._manual.pending = False
            self._board_canvas.set_pending_point(None)
            self._manual_maybe_analyze()
            return
        if not self._observe_button.isChecked():
            return
        self._tracker.reset()
        self._reset_move_history()
        self._set_board(BoardState.empty(), analyze=False)
        selected = self._my_color.currentText()
        self._candidate_text.setText("Color changed. Waiting for a stable board frame.")
        self._status.setText(f"Color changed to {selected}. Resynchronizing current board.")

    def _update_edit_mode(self) -> None:
        if self._input_mode is InputMode.MANUAL_RELAY:
            value = self._manual_setup_tool.currentData() if self._manual.setup else "relay"
        else:
            value = self._edit_mode.currentData()
        self._board_canvas.set_edit_mode(value)

    def _on_input_mode_changed(self, index: int) -> None:
        mode = InputMode.VISION if index == 0 else InputMode.MANUAL_RELAY
        if mode is self._input_mode:
            return
        if self._observe_button.isChecked():
            self._observe_button.setChecked(False)
        self._input_mode = mode
        self._manual.pending = False
        self._manual.changing_move = False
        self._board_canvas.set_pending_point(None)
        self._update_edit_mode()
        self._refresh_manual_controls()
        self._status.setText("已切换到手动搬运。" if mode is InputMode.MANUAL_RELAY else "已切换到自动识别。")

    def _refresh_manual_controls(self) -> None:
        active = self._input_mode is InputMode.MANUAL_RELAY
        self._manual_start_button.setEnabled(active)
        self._manual_continue_button.setEnabled(active and self._board.is_count_legal())
        self._manual_undo_button.setEnabled(active and bool(self._manual_history))
        self._manual_change_button.setEnabled(active and self._manual.pending)
        self._manual_setup_tool.setVisible(self._manual.setup)
        self._manual_finish_setup_button.setVisible(self._manual.setup)

    def start_manual_game(self) -> None:
        color = self._my_color.currentData()
        if not isinstance(color, Stone):
            self._status.setText("请先选择自己执黑或执白。")
            return
        self._clear_correction_state(record=False)
        self._tracker.reset()
        self._manual = ManualRelayState()
        self._manual_history.clear()
        self._reset_move_history(start_new_game=False)
        self._session.start_game(color, input_mode=InputMode.MANUAL_RELAY.value)
        self._set_board(BoardState.empty(), analyze=False)
        self._refresh_manual_controls()
        self._manual_maybe_analyze()

    def continue_manual_game(self) -> None:
        if not self._board.is_count_legal():
            self._status.setText("当前局面黑白棋数量不合法，无法继续。")
            return
        self._manual = ManualRelayState()
        self._manual_history.clear()
        self._session.start_game(
            self._my_color.currentData() if isinstance(self._my_color.currentData(), Stone) else None,
            input_mode=InputMode.MANUAL_RELAY.value,
        )
        self._manual_maybe_analyze()
        self._refresh_manual_controls()

    def _manual_place(self, x: int, y: int, source: str) -> bool:
        if not self._board.in_bounds(x, y) or self._board.at(x, y) is not Stone.EMPTY:
            self._status.setText("该交叉点已有棋子。")
            return False
        previous = self._board
        current = previous.place(x, y)
        self._manual_history.append(previous)
        self._record_observed_moves(previous, current, source=source)
        self._set_board(current, analyze=False)
        self._refresh_manual_controls()
        return True

    def _manual_maybe_analyze(self) -> None:
        color = self._my_color.currentData()
        if not isinstance(color, Stone):
            self._status.setText("请先选择自己执黑或执白。")
            return
        if self._board.is_terminal():
            self.analyze_current_board()
            return
        if self._board.side_to_move() is color:
            self._status.setText("正在为你分析下一手...")
            self.analyze_current_board()
        else:
            side = "黑棋" if self._board.side_to_move() is Stone.BLACK else "白棋"
            self._candidate_text.setText(f"请在左侧录入对方的{side}落子。")
            self._status.setText("等待对方落子。")

    def _on_manual_relay_point(self, x: int, y: int) -> None:
        if self._input_mode is not InputMode.MANUAL_RELAY or self._manual.setup:
            return
        if self._manual.changing_move:
            if self._manual_place(x, y, "manual_relay_override"):
                self._manual.changing_move = False
                self._manual.pending = True
                self._manual.pending_before = self._manual_history[-1]
                self._board_canvas.set_pending_point((x, y))
                self._candidate_text.setText(
                    f"已改走 {self._board.coordinate_name(x, y)}，请在真实棋盘落此点。"
                )
            return
        if self._manual.pending:
            if self._manual_place(x, y, "manual_relay_opponent"):
                self._manual.pending = False
                self._manual.pending_before = None
                self._board_canvas.set_pending_point(None)
                self._manual_maybe_analyze()
            return
        color = self._my_color.currentData()
        if not isinstance(color, Stone):
            self._status.setText("请先选择自己执黑或执白。")
            return
        source = "manual_relay_manual" if self._board.side_to_move() is color else "manual_relay_opponent"
        if not self._manual_place(x, y, source):
            return
        if source == "manual_relay_manual":
            self._manual.pending = True
            self._manual.pending_before = self._manual_history[-1]
            self._board_canvas.set_pending_point((x, y))
            self._candidate_text.setText(f"请在真实棋盘落 {self._board.coordinate_name(x, y)}。")
        else:
            self._manual_maybe_analyze()

    def change_manual_move(self) -> None:
        if not self._manual.pending or self._manual.pending_before is None:
            return
        self._session.remove_last_move()
        self._manual_history.pop()
        before = self._manual.pending_before
        self._manual.pending = False
        self._manual.changing_move = True
        self._manual.pending_before = None
        candidates = self._candidates
        self._set_board(before, analyze=False)
        self._candidates = candidates
        self._board_canvas.set_candidates(candidates)
        self._status.setText("点击左侧棋盘选择本手实际落点。")
        self._refresh_manual_controls()

    def undo_manual_move(self) -> None:
        if not self._manual_history:
            return
        previous = self._manual_history.pop()
        self._session.remove_last_move()
        self._manual = ManualRelayState()
        self._board_canvas.set_pending_point(None)
        self._set_board(previous, analyze=False)
        self._refresh_manual_controls()
        self._manual_maybe_analyze()

    def _toggle_manual_setup(self, enabled: bool) -> None:
        if self._input_mode is not InputMode.MANUAL_RELAY:
            return
        self._manual.setup = enabled
        self._manual.pending = False
        self._board_canvas.set_pending_point(None)
        self._update_edit_mode()
        self._refresh_manual_controls()

    def finish_manual_setup(self) -> None:
        if not self._board.is_count_legal():
            self._status.setText("局面编辑未完成：黑白棋数量不合法。")
            return
        self._manual.setup = False
        self._manual_setup_button.setChecked(False)
        self._reset_move_history()
        self._update_edit_mode()
        self._refresh_manual_controls()
        self._manual_maybe_analyze()

    def _on_manual_board_edit(self, board: BoardState) -> None:
        if self._input_mode is InputMode.MANUAL_RELAY and self._manual.setup:
            self._set_board(board, analyze=False)
            self._status.setText("正在编辑局面。完成后将检查合法性并开始分析。")
            return
        previous = self._board
        changed = [
            (x, y)
            for y in range(board.size)
            for x in range(board.size)
            if previous.at(x, y) is not board.at(x, y)
        ]
        if len(changed) != 1:
            return
        x, y = changed[0]
        event = self._corrections.set_cell(self._raw_board(), x, y, board.at(x, y))
        if event is None:
            self._board_canvas.set_board(previous)
            return
        corrected = self._corrections.apply(self._raw_board())
        self._record_correction(event)
        self._record_manual_change(previous, corrected)
        self._set_corrected_board(
            corrected,
            f"Manual correction: {corrected.coordinate_name(x, y)} {event.action}. "
            f"{self._corrections.count} correction(s) active.",
        )

    def undo_correction(self) -> None:
        event = self._corrections.undo(self._raw_board())
        if event is None:
            self._status.setText("No manual correction to undo.")
            return
        previous = self._board
        corrected = self._corrections.apply(self._raw_board())
        self._record_correction(event)
        self._record_manual_change(previous, corrected)
        self._set_corrected_board(
            corrected,
            f"Undid correction at {corrected.coordinate_name(event.x, event.y)}. "
            f"{self._corrections.count} correction(s) active.",
        )

    def clear_corrections(self) -> None:
        events = self._clear_correction_state(record=True)
        if not events:
            self._status.setText("No manual corrections are active.")
            return
        self._tracker.reset()
        self._raw_reset_samples.clear()
        self._invalidate_analysis()
        self._candidate_text.setText("Corrections cleared. Waiting for a stable visual board.")
        self._status.setText("Corrections cleared. Resynchronizing from stable visual frames.")

    def _set_board(self, board: BoardState, analyze: bool) -> None:
        self._invalidate_analysis()
        self._board = board
        self._board_canvas.set_board(board)
        if analyze:
            self.analyze_current_board()

    def _invalidate_analysis(self) -> None:
        self._analysis_version += 1
        self._candidates = ()
        self._board_canvas.set_candidates(())
        self._board_canvas.set_danger_points(())

    def clear_board(self) -> None:
        if self._observe_button.isChecked():
            self._tracker.reset()
            self._invalidate_analysis()
            black, white = self._board.counts()
            self._candidate_text.setText(
                f"Resynchronizing. Last confirmed board: B{black}/W{white}."
            )
            self._status.setText("Resynchronizing current board from stable frames.")
            return
        self._tracker.reset()
        self._reset_move_history(start_new_game=True)
        self._set_board(BoardState.empty(), analyze=False)
        self._candidate_text.setText("No analysis yet")
        self._status.setText("Board cleared.")

    def select_rapfi(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Rapfi.exe",
            str(Path.cwd()),
            "Rapfi executable (Rapfi.exe);;Executables (*.exe)",
        )
        if not filename:
            return
        self._invalidate_analysis()
        if self._rapfi_analyzer is not None:
            self._rapfi_analyzer.close()
            self._rapfi_analyzer = None
        self._rapfi_path = Path(filename)
        self._engine_status.setText(self._engine_status_text())
        self._status.setText(f"Rapfi selected: {self._rapfi_path.name}")

    def analyze_current_board(self) -> None:
        if not self._board.is_count_legal():
            self._status.setText("Cannot analyze: black/white counts are invalid.")
            return
        if self._board.is_terminal():
            self._session.finish_game(self._board)
            winner = self._board.winner()
            self._candidate_text.setText(
                "Game over: " + ("Black wins." if winner is Stone.BLACK else "White wins." if winner else "Draw.")
            )
            return

        board = self._board
        self._analysis_version += 1
        version = self._analysis_version
        config = self._current_rapfi_config()
        search_budget_ms = self._search_budget_ms(board)
        if config is not None:
            if self._rapfi_analyzer is None or self._rapfi_analyzer.config != config:
                if self._rapfi_analyzer is not None:
                    self._rapfi_analyzer.close()
                self._rapfi_analyzer = RapfiAnalyzer(config)
            analyzer: Any = self._rapfi_analyzer
            side = "Black" if board.side_to_move() is Stone.BLACK else "White"
            self._status.setText(
                f"Analyzing {side}: up to {search_budget_ms / 1000:.0f}s engine time, "
                f"{MAX_TOTAL_ANALYSIS_TIME_MS / 1000:.0f}s total limit."
            )
        else:
            analyzer = HeuristicAnalyzer()
            self._status.setText("Analyzing with local tactical heuristic...")

        self._pending_analyses += 1
        self._set_engine_controls_enabled(False)
        if isinstance(analyzer, RapfiAnalyzer):
            future = self._executor.submit(
                analyzer.analyze,
                board,
                3,
                time_ms=search_budget_ms,
                total_timeout_ms=MAX_TOTAL_ANALYSIS_TIME_MS,
            )
        else:
            future = self._executor.submit(analyzer.analyze, board, 3)

        def completed(task: concurrent.futures.Future[AnalysisResult]) -> None:
            try:
                result: object = task.result()
            except Exception as error:
                result = error
            self.analysis_ready.emit(version, result)
            self.analysis_finished.emit()

        future.add_done_callback(completed)

    def _on_analysis_finished(self) -> None:
        self._pending_analyses = max(0, self._pending_analyses - 1)
        if self._pending_analyses == 0 and not self._observation_waiting_for_engine:
            self._set_engine_controls_enabled(True)

    def _on_analysis_ready(self, version: int, payload: object) -> None:
        if version != self._analysis_version:
            return
        if isinstance(payload, Exception):
            self._candidate_text.setText(f"Engine error: {payload}")
            self._status.setText("Analysis failed. Falling back to the local heuristic is available.")
            return
        result = payload
        if not isinstance(result, AnalysisResult):
            return
        if result.board != self._board:
            return
        self._candidates = result.candidates
        self._board_canvas.set_candidates(result.candidates)
        self._board_canvas.set_danger_points(
            result.danger_points
            if result.recommendation_mode is RecommendationMode.FORCED_LOSS
            else ()
        )
        self._session.append(result)
        if not result.candidates:
            if result.recommendation_mode is RecommendationMode.FORCED_LOSS:
                points = ", ".join(
                    self._board.coordinate_name(x, y) for x, y in result.danger_points
                )
                self._candidate_text.setText(
                    "No single move prevents immediate loss."
                    + (f" Opponent winning points: {points}." if points else "")
                )
                self._status.setText("Immediate threats cannot all be blocked.")
            else:
                self._candidate_text.setText(f"{result.engine_name}: no safe legal move.")
            return
        color = self._my_color.currentData()
        if (
            self._input_mode is InputMode.MANUAL_RELAY
            and not self._manual.pending
            and not self._manual.changing_move
            and isinstance(color, Stone)
            and self._board.side_to_move() is color
        ):
            chosen = result.candidates[0]
            before = self._board
            if self._manual_place(chosen.x, chosen.y, "manual_relay_auto"):
                self._manual.pending = True
                self._manual.pending_before = before
                self._board_canvas.set_pending_point((chosen.x, chosen.y))
                self._candidates = result.candidates
                self._board_canvas.set_candidates(result.candidates)
                self._candidate_text.setText(
                    f"已自动采用第 1 推荐：{before.coordinate_name(chosen.x, chosen.y)}。"
                    "请在真实棋盘落此点；对方回应后再录入左侧棋盘。"
                )
                self._status.setText("等待对方回应，可在回应前点击“改走本手”。")
                self._refresh_manual_controls()
            return
        lines: list[str] = []
        if result.recommendation_mode is RecommendationMode.WIN_NOW:
            lines.append("Immediate win available.")
        elif result.recommendation_mode is RecommendationMode.FORCED_DEFENSE:
            move = result.candidates[0]
            lines.append(
                "Only move to avoid immediate loss: "
                + self._board.coordinate_name(move.x, move.y)
            )
        lines.extend(
            "\n".join(
                f"{move.rank}. {self._board.coordinate_name(move.x, move.y)}"
                f"  {self._candidate_label(move, result)}"
                f"  {move.proof.value}"
                for move in result.candidates
            ).splitlines()
        )
        lines.append(f"Engine: {result.engine_name}")
        if result.search_stats is not None:
            stats = result.search_stats
            details = [
                f"Search: {stats.elapsed_ms}ms / {stats.requested_time_ms}ms budget",
            ]
            if stats.engine_time_ms is not None:
                details.append(f"engine {stats.engine_time_ms}ms")
            if stats.depth is not None:
                details.append(f"depth {stats.depth}")
            if stats.threads is not None:
                details.append(f"{stats.threads} threads")
            if stats.hash_kib is not None:
                details.append(f"{stats.hash_kib // 1024}MB hash")
            lines.append("; ".join(details))
        self._candidate_text.setText("\n".join(lines))
        self._status.setText(f"Analysis ready from {result.engine_name}.")

    @staticmethod
    def _candidate_label(move: CandidateMove, result: AnalysisResult) -> str:
        if move.rank == 1 and result.engine_name.startswith("Rapfi"):
            return "Rapfi best"
        if move.rank > 1 and result.engine_name == "Rapfi + tactical alternatives":
            return "local alternative"
        return "engine choice" if move.score is None else f"score {move.score:+d}"

    def closeEvent(self, event: Any) -> None:
        self._capture_timer.stop()
        self._capture_session.stop()
        target = self._session.save()
        if self._rapfi_analyzer is not None:
            self._rapfi_analyzer.close()
        self._executor.shutdown(wait=False, cancel_futures=True)
        if target:
            self._status.setText(f"Saved session: {target.name}")
        event.accept()


def run() -> int:
    application = QApplication(sys.argv)
    application.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return application.exec()
