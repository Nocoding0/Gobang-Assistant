from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from ctypes import wintypes


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


if os.name == "nt":
    user32 = ctypes.windll.user32


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
        rect = wintypes.RECT()
        if not user32.GetWindowRect(handle, ctypes.byref(rect)):
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
    rect = wintypes.RECT()
    if not user32.GetWindowRect(handle, ctypes.byref(rect)):
        return None
    return WindowInfo(
        handle=handle,
        title=buffer.value.strip(),
        left=rect.left,
        top=rect.top,
        right=rect.right,
        bottom=rect.bottom,
    )
