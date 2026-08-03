import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QDialogButtonBox

from gomoku_assistant.analysis import AnalysisResult, CandidateMove, ProofStatus, RecommendationMode
from gomoku_assistant.domain import BoardState, Stone
from gomoku_assistant.ui import CalibrationDialog, MainWindow
from gomoku_assistant.vision import CellEvidence, RecognitionResult


def _recognition(board: BoardState) -> RecognitionResult:
    evidence = tuple(
        CellEvidence(
            black=1.0 if stone is Stone.BLACK else 0.0,
            white=1.0 if stone is Stone.WHITE else 0.0,
            empty=1.0 if stone is Stone.EMPTY else 0.0,
        )
        for stone in board.cells
    )
    return RecognitionResult(
        board=board,
        confidence=1.0,
        cell_confidences=(1.0,) * len(board.cells),
        board_visible=True,
        grid_score=1.0,
        warped=np.zeros((841, 841, 3), dtype=np.uint8),
        cell_evidence=evidence,
    )


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


def test_forced_loss_marks_dangers_without_inventing_candidates() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    board = BoardState.empty().set_cell(7, 7, Stone.BLACK)
    window._board = board
    window._board_canvas.set_board(board)
    window._analysis_version = 1

    window._on_analysis_ready(
        1,
        AnalysisResult(
            board=board,
            candidates=(),
            engine_name="Tactical safety check",
            recommendation_mode=RecommendationMode.FORCED_LOSS,
            danger_points=((6, 7), (8, 7)),
        ),
    )

    assert application is not None
    assert window._candidates == ()
    assert window._board_canvas._danger_points == ((6, 7), (8, 7))
    assert "No single move prevents immediate loss" in window._candidate_text.text()
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


def test_manual_correction_survives_a_repeated_missing_stone_while_observing() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    raw = BoardState.empty()
    window._board = raw
    window._board_canvas.set_board(raw)
    window._last_raw_recognition = _recognition(raw)

    window._on_manual_board_edit(raw.set_cell(7, 7, Stone.BLACK))

    assert application is not None
    assert window._board.at(7, 7) is Stone.BLACK
    assert window._corrections.overrides == {(7, 7): Stone.BLACK}

    raw_after_white = raw.set_cell(8, 7, Stone.WHITE)
    corrected = window._apply_corrections(_recognition(raw_after_white))
    for _ in range(3):
        _, committed = window._tracker.observe(corrected)

    assert committed is not None
    assert committed.counts() == (1, 1)
    assert committed.at(7, 7) is Stone.BLACK
    assert committed.at(8, 7) is Stone.WHITE
    window.close()


def test_manual_erase_persists_and_can_be_undone() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    raw = BoardState.empty().set_cell(7, 7, Stone.BLACK)
    window._board = raw
    window._board_canvas.set_board(raw)
    window._last_raw_recognition = _recognition(raw)

    window._on_manual_board_edit(raw.set_cell(7, 7, Stone.EMPTY))

    assert application is not None
    assert window._board.at(7, 7) is Stone.EMPTY
    assert window._corrections.overrides == {(7, 7): Stone.EMPTY}

    window.undo_correction()

    assert window._board.at(7, 7) is Stone.BLACK
    assert window._corrections.count == 0
    window.close()


def test_stable_raw_new_game_clears_persistent_manual_corrections() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    raw = BoardState.empty()
    window._board = raw
    window._board_canvas.set_board(raw)
    window._last_raw_recognition = _recognition(raw)
    window._on_manual_board_edit(raw.set_cell(7, 7, Stone.BLACK))

    first = window._maybe_start_new_game_from_raw_frame(_recognition(raw))
    second = window._maybe_start_new_game_from_raw_frame(_recognition(raw))

    assert application is not None
    assert not first
    assert second
    assert window._corrections.count == 0
    assert window._board == BoardState.empty()
    window.close()
