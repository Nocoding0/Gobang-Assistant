import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QDialogButtonBox

from gomoku_assistant.ui import CalibrationDialog


def test_low_contrast_calibration_allows_save() -> None:
    application = QApplication.instance() or QApplication([])
    frame = np.full((841, 841, 3), (188, 178, 150), dtype=np.uint8)
    for coordinate in range(0, 841, 60):
        cv2.line(frame, (coordinate, 0), (coordinate, 840), (195, 168, 138), 1)
        cv2.line(frame, (0, coordinate), (840, coordinate), (195, 168, 138), 1)

    dialog = CalibrationDialog(frame)
    dialog._canvas._points = [(0.0, 0.0), (840.0, 0.0), (840.0, 840.0), (0.0, 840.0)]
    dialog._on_points_changed(4)
    save_button = dialog._buttons.button(QDialogButtonBox.StandardButton.Save)

    assert application is not None
    assert save_button.isEnabled()
    assert "Low-contrast" in dialog._hint.text()
