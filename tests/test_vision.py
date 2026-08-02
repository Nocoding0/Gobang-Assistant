import cv2
import numpy as np

from gomoku_assistant.domain import BoardState, Stone
from gomoku_assistant.vision import (
    BoardProfile,
    RecognitionResult,
    StableStateTracker,
    recognize_frame,
)


def _synthetic_board() -> tuple[np.ndarray, BoardProfile]:
    size = 15
    edge = 840
    canvas = np.full((edge + 1, edge + 1, 3), (95, 170, 220), dtype=np.uint8)
    for coordinate in range(0, edge + 1, 60):
        cv2.line(canvas, (coordinate, 0), (coordinate, edge), (45, 85, 105), 2)
        cv2.line(canvas, (0, coordinate), (edge, coordinate), (45, 85, 105), 2)
    cv2.circle(canvas, (7 * 60, 7 * 60), 25, (20, 20, 20), -1)
    cv2.circle(canvas, (8 * 60, 7 * 60), 25, (250, 250, 250), -1)
    profile = BoardProfile(
        board_size=size,
        corners=((0, 0), (edge, 0), (edge, edge), (0, edge)),
    )
    return canvas, profile


def test_recognizes_synthetic_black_and_white_stones() -> None:
    frame, profile = _synthetic_board()

    result = recognize_frame(frame, profile)

    assert result.board.at(7, 7) is Stone.BLACK
    assert result.board.at(8, 7) is Stone.WHITE
    assert result.board.at(0, 0) is Stone.EMPTY
    assert result.board_visible
    assert result.grid_score >= 0.35


def test_recognizes_black_stone_with_colored_last_move_marker() -> None:
    frame, profile = _synthetic_board()
    cv2.circle(frame, (7 * 60, 7 * 60), 10, (0, 140, 255), -1)

    result = recognize_frame(frame, profile)

    assert result.board.at(7, 7) is Stone.BLACK


def test_rejects_non_board_image() -> None:
    frame = np.full((841, 841, 3), (130, 180, 210), dtype=np.uint8)
    profile = BoardProfile(
        board_size=15,
        corners=((0, 0), (840, 0), (840, 840), (0, 840)),
    )

    result = recognize_frame(frame, profile)

    assert not result.board_visible


def test_state_tracker_requires_stable_frames() -> None:
    frame, profile = _synthetic_board()
    recognition = recognize_frame(frame, profile)
    tracker = StableStateTracker(required_frames=3, min_confidence=0.1)

    _, first = tracker.observe(recognition)
    _, second = tracker.observe(recognition)
    transition, third = tracker.observe(recognition)

    assert first is None
    assert second is None
    assert third is not None
    assert transition.valid


def test_state_tracker_rejects_menu_without_board_grid() -> None:
    result = RecognitionResult(
        board=BoardState.empty(),
        confidence=1.0,
        cell_confidences=(1.0,) * 225,
        board_visible=False,
        grid_score=0.0,
        warped=np.zeros((841, 841, 3), dtype=np.uint8),
    )
    tracker = StableStateTracker()

    transition, committed = tracker.observe(result)

    assert not transition.valid
    assert "grid" in transition.reason
    assert committed is None


def test_state_tracker_starts_new_game_after_stable_reset() -> None:
    frame, profile = _synthetic_board()
    initial = recognize_frame(frame, profile)
    empty = RecognitionResult(
        board=BoardState.empty(),
        confidence=1.0,
        cell_confidences=(1.0,) * 225,
        board_visible=True,
        grid_score=1.0,
        warped=frame,
    )
    tracker = StableStateTracker(required_frames=1, reset_frames=2)

    _, committed_initial = tracker.observe(initial)
    first_reset, first_board = tracker.observe(empty)
    second_reset, second_board = tracker.observe(empty)

    assert committed_initial is not None
    assert not first_reset.valid
    assert first_board is None
    assert second_reset.valid
    assert second_reset.reason == "new game detected"
    assert second_board == BoardState.empty()


def test_state_tracker_commits_multi_move_catch_up() -> None:
    before = BoardState.empty().set_cell(7, 7, Stone.BLACK)
    after = before.set_cell(8, 7, Stone.WHITE).set_cell(9, 7, Stone.BLACK)
    warped = np.zeros((841, 841, 3), dtype=np.uint8)
    tracker = StableStateTracker(required_frames=1)

    _, first = tracker.observe(
        RecognitionResult(
            board=before,
            confidence=1.0,
            cell_confidences=(1.0,) * 225,
            board_visible=True,
            grid_score=1.0,
            warped=warped,
        )
    )
    transition, caught_up = tracker.observe(
        RecognitionResult(
            board=after,
            confidence=1.0,
            cell_confidences=(1.0,) * 225,
            board_visible=True,
            grid_score=1.0,
            warped=warped,
        )
    )

    assert first == before
    assert transition.valid
    assert transition.added_count == 2
    assert caught_up == after
