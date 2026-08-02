from __future__ import annotations

import concurrent.futures
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
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
    QVBoxLayout,
    QWidget,
)

from .analysis import AnalysisResult, CandidateMove, HeuristicAnalyzer
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
from .domain import BoardState, Stone
from .engine import RapfiAnalyzer, RapfiConfig
from .sessions import SessionLogger
from .vision import BoardProfile, StableStateTracker, grid_points_in_source, order_corners, recognize_frame


MARKER_COLORS = (QColor("#c73b33"), QColor("#c88810"), QColor("#167e79"))


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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(520, 520)
        self._board = BoardState.empty()
        self._candidates: tuple[CandidateMove, ...] = ()
        self._edit_mode: Stone | None = None

    @property
    def board(self) -> BoardState:
        return self._board

    def set_board(self, board: BoardState) -> None:
        self._board = board
        self.update()

    def set_candidates(self, candidates: tuple[CandidateMove, ...]) -> None:
        self._candidates = candidates
        self.update()

    def set_edit_mode(self, stone: Stone | None) -> None:
        self._edit_mode = stone

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
        current = self._board.at(x, y)
        if self._edit_mode is Stone.EMPTY:
            next_board = self._board.set_cell(x, y, Stone.EMPTY)
        elif self._edit_mode is not None:
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

        line = self._board.winning_line()
        if line:
            painter.setPen(QPen(QColor("#267d4b"), max(3.0, spacing * 0.09)))
            painter.drawLine(
                self._point_to_screen(*line.points[0]), self._point_to_screen(*line.points[-1])
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
        self._save_button.setEnabled(count == 4)

    def profile(self) -> BoardProfile:
        if len(self._canvas.points) != 4:
            raise ValueError("Four calibration points are required.")
        return BoardProfile(board_size=15, corners=order_corners(self._canvas.points))


class SuggestionOverlay(QWidget):
    def __init__(self) -> None:
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
        )
        super().__init__(None, flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._markers: list[tuple[float, float, CandidateMove]] = []

    def update_markers(
        self,
        window: WindowInfo,
        frame_shape: tuple[int, int],
        profile: BoardProfile,
        candidates: tuple[CandidateMove, ...],
    ) -> None:
        frame_height, frame_width = frame_shape
        source_points = grid_points_in_source(profile)
        markers: list[tuple[float, float, CandidateMove]] = []
        for candidate in candidates:
            index = candidate.y * profile.board_size + candidate.x
            source_x, source_y = source_points[index]
            markers.append((source_x / frame_width, source_y / frame_height, candidate))
        self._markers = markers
        self.setGeometry(QRect(window.left, window.top, window.width, window.height))
        self.update()

    def paintEvent(self, _: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        radius = max(min(self.width(), self.height()) * 0.022, 14)
        for normalized_x, normalized_y, candidate in self._markers:
            center = QPointF(normalized_x * self.width(), normalized_y * self.height())
            color = MARKER_COLORS[min(candidate.rank - 1, len(MARKER_COLORS) - 1)]
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(color, 4))
            painter.drawEllipse(center, radius, radius)
            painter.setBrush(color)
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawText(
                QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2),
                Qt.AlignmentFlag.AlignCenter,
                str(candidate.rank),
            )


class MainWindow(QMainWindow):
    analysis_ready = Signal(int, object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Gomoku Training Assistant")
        self.resize(1220, 820)
        self._board = BoardState.empty()
        self._candidates: tuple[CandidateMove, ...] = ()
        self._profile: BoardProfile | None = self._load_profile()
        self._last_frame: np.ndarray | None = None
        self._last_window: WindowInfo | None = None
        self._capture_session = WindowCaptureSession()
        self._tracker = StableStateTracker()
        self._overlay = SuggestionOverlay()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._analysis_version = 0
        self._session = SessionLogger(Path.cwd() / "sessions")
        self._rapfi_path = self._default_rapfi_path()
        self._rapfi_analyzer: RapfiAnalyzer | None = None

        self._board_canvas = BoardCanvas()
        self._board_canvas.board_edited.connect(self._on_manual_board_edit)
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

        self._search_time = QSpinBox()
        self._search_time.setRange(100, 10_000)
        self._search_time.setSingleStep(100)
        self._search_time.setValue(1000)
        self._edit_mode = QComboBox()
        self._edit_mode.addItem("Auto move", None)
        self._edit_mode.addItem("Place black", Stone.BLACK)
        self._edit_mode.addItem("Place white", Stone.WHITE)
        self._edit_mode.addItem("Erase", Stone.EMPTY)
        self._edit_mode.currentIndexChanged.connect(self._update_edit_mode)
        self._turn_scope = QComboBox()
        self._turn_scope.addItem("Analyze both colors", None)
        self._turn_scope.addItem("Only black turns", Stone.BLACK)
        self._turn_scope.addItem("Only white turns", Stone.WHITE)
        self._overlay_toggle = QCheckBox("Show transparent overlay")
        self._overlay_toggle.setChecked(True)
        self._overlay_toggle.toggled.connect(self._refresh_overlay)

        self._profile_status = QLabel(self._profile_status_text())
        self._engine_status = QLabel(self._engine_status_text())
        self._status = QLabel("Ready")
        self._candidate_text = QLabel("No analysis yet")
        self._candidate_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._candidate_text.setWordWrap(True)

        control_group = QGroupBox("Capture and analysis")
        controls = QFormLayout(control_group)
        controls.addRow("Target window", self._window_combo)
        controls.addRow("", self._refresh_windows_button)
        controls.addRow("", self._capture_button)
        controls.addRow("", self._load_image_button)
        controls.addRow("", self._calibrate_button)
        controls.addRow("Profile", self._profile_status)
        controls.addRow("", self._observe_button)
        controls.addRow("Edit board", self._edit_mode)
        controls.addRow("Analyze turns", self._turn_scope)
        controls.addRow("Rapfi search", self._search_time)
        controls.addRow("", self._engine_button)
        controls.addRow("Engine", self._engine_status)
        controls.addRow("", self._analyze_button)
        controls.addRow("", self._clear_button)
        controls.addRow("", self._overlay_toggle)

        suggestion_group = QGroupBox("Suggestions")
        suggestion_layout = QVBoxLayout(suggestion_group)
        suggestion_layout.addWidget(self._candidate_text)
        suggestion_layout.addStretch(1)

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.addWidget(control_group)
        right_layout.addWidget(suggestion_group)
        right_layout.addWidget(QLabel("Latest source frame"))
        right_layout.addWidget(self._preview, 1)
        right_layout.addWidget(self._status)

        splitter = QSplitter()
        splitter.addWidget(self._board_canvas)
        splitter.addWidget(right_column)
        splitter.setSizes([760, 420])
        self.setCentralWidget(splitter)

        self._capture_timer = QTimer(self)
        self._capture_timer.setInterval(250)
        self._capture_timer.timeout.connect(self._observe_tick)
        self.analysis_ready.connect(self._on_analysis_ready)
        self.refresh_windows()
        self._update_edit_mode()

    def _default_rapfi_path(self) -> Path | None:
        candidate = Path.cwd() / "vendor" / "rapfi" / "Rapfi.exe"
        return candidate if candidate.is_file() else None

    def _profile_path(self) -> Path:
        return Path.cwd() / "profiles" / "default.json"

    def _load_profile(self) -> BoardProfile | None:
        path = self._profile_path()
        if not path.is_file():
            return None
        try:
            return BoardProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _save_profile(self) -> None:
        if self._profile is None:
            return
        path = self._profile_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._profile.to_dict(), indent=2), encoding="utf-8")

    def _profile_status_text(self) -> str:
        if self._profile is None:
            return "Not calibrated"
        if self._profile.source_width is None:
            return "Legacy profile: recalibrate"
        return "15x15 calibrated"

    def _engine_status_text(self) -> str:
        if self._rapfi_path and self._rapfi_path.is_file():
            return self._rapfi_path.name
        return "Local tactical heuristic"

    def _selected_handle(self) -> int | None:
        data = self._window_combo.currentData()
        return int(data) if data is not None else None

    def refresh_windows(self) -> None:
        current = self._selected_handle()
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
        if self._window_combo.count() == 0:
            self._status.setText("No eligible Windows application window found.")

    def _on_target_window_changed(self, _: int) -> None:
        self._capture_session.stop()
        self._last_frame = None
        self._last_window = None
        self._tracker.reset()
        self._overlay.hide()
        self._preview.setPixmap(QPixmap())
        self._preview.setText("No frame captured")

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
        self._save_profile()
        self._profile_status.setText(self._profile_status_text())
        self._status.setText("Calibration saved. Start observation or capture a new frame.")

    def _toggle_observation(self, enabled: bool) -> None:
        if enabled and self._profile is None:
            self._observe_button.setChecked(False)
            QMessageBox.information(self, "Calibration required", "Calibrate the 15x15 board first.")
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
        self._observe_button.setText("Stop observing" if enabled else "Start observing")
        if enabled:
            self._tracker.reset()
            self._capture_timer.start()
            self._status.setText("Observing target window.")
        else:
            self._capture_timer.stop()
            self._overlay.hide()
            self._status.setText("Observation paused.")

    def _observe_tick(self) -> None:
        if self._profile is None:
            return
        frame = self.capture_frame()
        if frame is None:
            return
        try:
            recognition = recognize_frame(frame, self._profile)
            transition, committed = self._tracker.observe(recognition)
        except Exception as error:
            self._status.setText(f"Recognition paused: {error}")
            return
        if committed is not None:
            self._set_board(committed, analyze=self._should_analyze_turn(committed))
            self._status.setText(
                f"Committed visual board: {transition.reason}; confidence {recognition.confidence:.0%}"
            )
        elif not transition.valid:
            self._overlay.hide()
            self._status.setText(f"Recognition paused: {transition.reason}")

    def _should_analyze_turn(self, board: BoardState) -> bool:
        scope = self._turn_scope.currentData()
        return scope is None or board.side_to_move() is scope

    def _update_edit_mode(self) -> None:
        value = self._edit_mode.currentData()
        self._board_canvas.set_edit_mode(value)

    def _on_manual_board_edit(self, board: BoardState) -> None:
        self._set_board(board, analyze=False)
        self._status.setText("Manual board edited. Select Analyze position when ready.")

    def _set_board(self, board: BoardState, analyze: bool) -> None:
        self._board = board
        self._board_canvas.set_board(board)
        self._candidates = ()
        self._board_canvas.set_candidates(())
        self._overlay.hide()
        if analyze:
            self.analyze_current_board()

    def clear_board(self) -> None:
        self._tracker.reset()
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
            winner = self._board.winner()
            self._candidate_text.setText(
                "Game over: " + ("Black wins." if winner is Stone.BLACK else "White wins." if winner else "Draw.")
            )
            self._overlay.hide()
            return

        board = self._board
        self._analysis_version += 1
        version = self._analysis_version
        self._status.setText("Analyzing...")
        if self._rapfi_path and self._rapfi_path.is_file():
            config = RapfiConfig(executable=self._rapfi_path, time_ms=self._search_time.value())
            if self._rapfi_analyzer is None or self._rapfi_analyzer.config != config:
                if self._rapfi_analyzer is not None:
                    self._rapfi_analyzer.close()
                self._rapfi_analyzer = RapfiAnalyzer(config)
            analyzer: Any = self._rapfi_analyzer
        else:
            analyzer = HeuristicAnalyzer()

        future = self._executor.submit(analyzer.analyze, board, 3)

        def completed(task: concurrent.futures.Future[AnalysisResult]) -> None:
            try:
                result: object = task.result()
            except Exception as error:
                result = error
            self.analysis_ready.emit(version, result)

        future.add_done_callback(completed)

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
        self._candidates = result.candidates
        self._board_canvas.set_candidates(result.candidates)
        self._session.append(self._board, result)
        if not result.candidates:
            self._candidate_text.setText(f"{result.engine_name}: no legal move.")
            return
        self._candidate_text.setText(
            "\n".join(
                f"{move.rank}. {self._board.coordinate_name(move.x, move.y)}"
                f"  {'engine choice' if move.score is None else f'score {move.score:+d}'}"
                f"  {move.proof.value}"
                for move in result.candidates
            )
            + f"\nEngine: {result.engine_name}"
        )
        self._status.setText(f"Analysis ready from {result.engine_name}.")
        self._refresh_overlay()

    def _refresh_overlay(self) -> None:
        if (
            not self._overlay_toggle.isChecked()
            or not self._candidates
            or self._profile is None
            or self._last_frame is None
        ):
            self._overlay.hide()
            return
        handle = self._selected_handle()
        window = get_window_info(handle) if handle is not None else self._last_window
        if window is None:
            self._overlay.hide()
            return
        self._overlay.update_markers(
            window,
            self._last_frame.shape[:2],
            self._profile,
            self._candidates,
        )
        self._overlay.show()

    def closeEvent(self, event: Any) -> None:
        self._capture_timer.stop()
        self._capture_session.stop()
        self._overlay.hide()
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
