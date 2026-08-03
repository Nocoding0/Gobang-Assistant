import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QDialogButtonBox

from gomoku_assistant.analysis import AnalysisResult, CandidateMove, ProofStatus
from gomoku_assistant.domain import BoardState, Stone
from gomoku_assistant.ui import CalibrationDialog, MainWindow


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


def test_stale_analysis_result_does_not_replace_new_board() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    old_board = BoardState.empty().set_cell(7, 7, Stone.BLACK)
    current_board = old_board.set_cell(7, 6, Stone.WHITE)
    stale = AnalysisResult(
        board=old_board,
        candidates=(
            CandidateMove(
                x=6,
                y=7,
                rank=1,
                score=100,
                proof=ProofStatus.HEURISTIC,
            ),
        ),
        engine_name="Rapfi",
    )
    window._board = current_board
    window._analysis_version = 4

    window._on_analysis_ready(4, stale)

    assert application is not None
    assert window._candidates == ()
    window.close()


def test_suggestions_stay_inside_assistant_window() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    board = BoardState.empty().set_cell(7, 7, Stone.BLACK)
    candidate = CandidateMove(
        x=8,
        y=7,
        rank=1,
        score=100,
        proof=ProofStatus.HEURISTIC,
    )

    window._board = board
    window._board_canvas.set_board(board)
    window._analysis_version = 1
    window._on_analysis_ready(
        1,
        AnalysisResult(board=board, candidates=(candidate,), engine_name="Rapfi"),
    )

    assert application is not None
    assert not hasattr(window, "_overlay")
    assert not hasattr(window, "_overlay_toggle")
    assert window._candidates == (candidate,)
    window.close()


def test_switching_to_white_while_observing_resynchronizes_board() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._board = BoardState.empty().set_cell(7, 7, Stone.BLACK)
    window._tracker.committed = window._board
    window._observe_button.blockSignals(True)
    window._observe_button.setChecked(True)
    window._observe_button.blockSignals(False)

    window._my_color.setCurrentIndex(2)

    assert application is not None
    assert window._board == BoardState.empty()
    assert window._tracker.committed is None
    assert "Resynchronizing" in window._status.text()
    window.close()


def test_resync_keeps_last_confirmed_board_while_observing() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    board = BoardState.empty().set_cell(7, 7, Stone.BLACK)
    window._board = board
    window._board_canvas.set_board(board)
    window._tracker.committed = board
    window._observe_button.blockSignals(True)
    window._observe_button.setChecked(True)
    window._observe_button.blockSignals(False)

    window.clear_board()

    assert application is not None
    assert window._board == board
    assert window._tracker.committed is None
    assert "Resynchronizing" in window._status.text()
    window.close()


def test_search_controls_cannot_exceed_fifteen_engine_seconds() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    original_black = window._black_search_time.value()
    original_white = window._white_search_time.value()

    window._black_search_time.setValue(99)
    window._white_search_time.setValue(99)

    assert application is not None
    assert window._black_search_time.value() == 15
    assert window._white_search_time.value() == 15
    window._black_search_time.setValue(original_black)
    window._white_search_time.setValue(original_white)
    window.close()


def test_observed_move_numbers_appear_on_the_assistant_board() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    previous = BoardState.empty()
    current = previous.place(7, 7, Stone.BLACK)

    window._record_observed_moves(previous, current, source="test")
    window._set_board(current, analyze=False)

    assert application is not None
    assert window._move_numbers == {(7, 7): 1}
    assert window._board_canvas._move_numbers == {(7, 7): 1}
    window.close()
