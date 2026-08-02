from __future__ import annotations

import ctypes
import os
import threading
import time
from dataclasses import dataclass
from ctypes import wintypes

import cv2
import numpy as np

try:
    from windows_capture import CaptureControl, Frame, WindowsCapture
except ImportError:
    CaptureControl = Frame = WindowsCapture = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class WindowInfo:
    handle: int
    title: str
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)


@dataclass(frozen=True)
class CapturedFrame:
    frame_bgr: np.ndarray
    window: WindowInfo
    captured_at: float


class CaptureError(RuntimeError):
    """Base error for window capture failures."""


class CaptureUnavailableError(CaptureError):
    """Windows Graphics Capture is not available in this Python environment."""


class CaptureTimeoutError(CaptureError):
    """A capture session did not produce an initial frame in time."""


class CaptureClosedError(CaptureError):
    """The selected capture target was closed."""


class BlankFrameError(CaptureError):
    """The capture backend returned a blank or nearly blank frame."""


if os.name == "nt":
    user32 = ctypes.windll.user32
    dwmapi = ctypes.windll.dwmapi


DWMWA_EXTENDED_FRAME_BOUNDS = 9


def is_blank_frame(frame_bgr: np.ndarray) -> bool:
    """Reject empty, malformed, and nearly all-black capture output."""

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3 or not frame_bgr.size:
        return True
    gray = cv2.cvtColor(frame_bgr[:, :, :3], cv2.COLOR_BGR2GRAY)
    return bool(np.percentile(gray, 99) <= 6 and float(gray.std()) <= 1.5)


class WindowCaptureSession:
    """Keeps one Graphics Capture session alive for the selected HWND."""

    def __init__(self, minimum_update_interval_ms: int = 100) -> None:
        self._minimum_update_interval_ms = minimum_update_interval_ms
        self._capture: object | None = None
        self._control: object | None = None
        self._target: WindowInfo | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_at = 0.0
        self._closed_error: CaptureError | None = None
        self._frame_ready = threading.Event()
        self._lock = threading.Lock()
        self._generation = 0

    @property
    def target_handle(self) -> int | None:
        return self._target.handle if self._target else None

    def start(self, window: WindowInfo) -> None:
        if self.target_handle == window.handle and self._control is not None:
            return
        self.stop()
        if WindowsCapture is None:
            raise CaptureUnavailableError(
                "windows-capture is not installed. Reinstall project dependencies."
            )

        with self._lock:
            self._generation += 1
            generation = self._generation
            self._target = window
            self._latest_frame = None
            self._latest_at = 0.0
            self._closed_error = None
            self._frame_ready.clear()
        capture = WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            secondary_window=False,
            minimum_update_interval=self._minimum_update_interval_ms,
            dirty_region=False,
            monitor_index=None,
            window_name=None,
            window_hwnd=window.handle,
        )

        @capture.event
        def on_frame_arrived(frame: Frame, _: object) -> None:
            try:
                copied = np.ascontiguousarray(frame.convert_to_bgr().frame_buffer).copy()
                with self._lock:
                    if generation != self._generation:
                        return
                    self._latest_frame = copied
                    self._latest_at = time.monotonic()
                    self._closed_error = None
                self._frame_ready.set()
            except Exception as error:
                with self._lock:
                    self._closed_error = CaptureError(f"Could not copy capture frame: {error}")
                self._frame_ready.set()

        @capture.event
        def on_closed() -> None:
            with self._lock:
                if generation != self._generation:
                    return
                self._closed_error = CaptureClosedError(
                    f"Target window closed: {window.title}"
                )
            self._frame_ready.set()

        self._capture = capture
        try:
            self._control = capture.start_free_threaded()
        except Exception as error:
            self._capture = None
            self._target = None
            raise CaptureError(f"Could not start Windows Graphics Capture: {error}") from error

    def latest_frame(self, window: WindowInfo, timeout_s: float = 1.5) -> CapturedFrame:
        self.start(window)
        if not self._frame_ready.wait(timeout_s):
            raise CaptureTimeoutError(
                f"No frame arrived from {window.title} within {timeout_s:.1f} seconds."
            )
        with self._lock:
            error = self._closed_error
            frame = self._latest_frame
            captured_at = self._latest_at
        if error is not None:
            raise error
        if frame is None:
            raise CaptureTimeoutError(f"Capture returned no frame for {window.title}.")
        if is_blank_frame(frame):
            raise BlankFrameError(
                "The selected window returned a blank frame. "
                "The app will not open calibration with this image."
            )
        return CapturedFrame(frame_bgr=frame.copy(), window=window, captured_at=captured_at)

    def stop(self) -> None:
        with self._lock:
            self._generation += 1
            control = self._control
            self._capture = None
            self._control = None
            self._target = None
            self._latest_frame = None
            self._latest_at = 0.0
            self._closed_error = None
            self._frame_ready.clear()
        if control is None:
            return
        try:
            control.stop()
            control.wait()
        except Exception:
            pass


def _window_rect(handle: int) -> wintypes.RECT | None:
    rect = wintypes.RECT()
    if os.name == "nt":
        result = dwmapi.DwmGetWindowAttribute(
            handle,
            DWMWA_EXTENDED_FRAME_BOUNDS,
            ctypes.byref(rect),
            ctypes.sizeof(rect),
        )
        if result == 0:
            return rect
    if user32.GetWindowRect(handle, ctypes.byref(rect)):
        return rect
    return None


def list_visible_windows(exclude_handle: int | None = None) -> tuple[WindowInfo, ...]:
    if os.name != "nt":
        return ()

    windows: list[WindowInfo] = []
    enum_callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def callback(handle: int, _: int) -> bool:
        if exclude_handle and handle == exclude_handle:
            return True
        if not user32.IsWindowVisible(handle):
            return True
        length = user32.GetWindowTextLengthW(handle)
        if length == 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(handle, buffer, length + 1)
        rect = _window_rect(handle)
        if rect is None:
            return True
        if rect.right - rect.left < 200 or rect.bottom - rect.top < 200:
            return True
        windows.append(
            WindowInfo(
                handle=int(handle),
                title=buffer.value.strip(),
                left=rect.left,
                top=rect.top,
                right=rect.right,
                bottom=rect.bottom,
            )
        )
        return True

    user32.EnumWindows(enum_callback_type(callback), 0)
    return tuple(sorted(windows, key=lambda item: item.title.casefold()))


def get_window_info(handle: int) -> WindowInfo | None:
    if os.name != "nt" or not user32.IsWindow(handle):
        return None
    length = user32.GetWindowTextLengthW(handle)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(handle, buffer, length + 1)
    rect = _window_rect(handle)
    if rect is None:
        return None
    return WindowInfo(
        handle=handle,
        title=buffer.value.strip(),
        left=rect.left,
        top=rect.top,
        right=rect.right,
        bottom=rect.bottom,
    )
