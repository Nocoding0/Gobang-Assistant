import cv2
import numpy as np

from gomoku_assistant.domain import BoardState, Stone
from gomoku_assistant.vision import (
    BoardProfile,
    RecognitionResult,
    StableStateTracker,
    is_valid_board_quadrilateral,
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


def _blue_board_with_stars() -> tuple[np.ndarray, BoardProfile]:
    edge = 840
    canvas = np.full((edge + 1, edge + 1, 3), (180, 142, 98), dtype=np.uint8)
    for coordinate in range(0, edge + 1, 60):
        cv2.line(canvas, (coordinate, 0), (coordinate, edge), (90, 56, 72), 2)
        cv2.line(canvas, (0, coordinate), (edge, coordinate), (90, 56, 72), 2)
    for x, y in ((3, 3), (11, 3), (3, 11), (11, 11)):
        cv2.circle(canvas, (x * 60, y * 60), 10, (70, 42, 58), -1)
    profile = BoardProfile(
        board_size=15,
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
    assert result.confidence >= 0.70


def test_recognizes_black_stone_with_colored_last_move_marker() -> None:
    frame, profile = _synthetic_board()
    cv2.circle(frame, (7 * 60, 7 * 60), 10, (0, 140, 255), -1)

    result = recognize_frame(frame, profile)

    assert result.board.at(7, 7) is Stone.BLACK


def test_recognizes_white_stone_with_colored_last_move_marker() -> None:
    frame, profile = _synthetic_board()
    cv2.circle(frame, (8 * 60, 7 * 60), 10, (0, 140, 255), -1)

    result = recognize_frame(frame, profile)

    assert result.board.at(8, 7) is Stone.WHITE


def test_ignores_small_blue_board_star_points() -> None:
    frame, profile = _blue_board_with_stars()
    cv2.circle(frame, (7 * 60, 7 * 60), 25, (20, 20, 20), -1)
    cv2.circle(frame, (8 * 60, 7 * 60), 25, (250, 250, 250), -1)

    result = recognize_frame(frame, profile)

    assert result.board.at(7, 7) is Stone.BLACK
    assert result.board.at(8, 7) is Stone.WHITE
    assert result.board.counts() == (1, 1)
    assert all(result.board.at(x, y) is Stone.EMPTY for x, y in ((3, 3), (11, 3), (3, 11), (11, 11)))
    assert result.confidence >= 0.70


def test_detects_large_banner_covering_board() -> None:
    frame, profile = _synthetic_board()
    cv2.rectangle(frame, (135, 580), (710, 665), (20, 20, 20), -1)
    cv2.putText(
        frame,
        "Disconnected",
        (220, 635),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (250, 250, 250),
        2,
        cv2.LINE_AA,
    )

    result = recognize_frame(frame, profile)

    assert result.obstruction_reason == "board is covered by a popup or banner"


def test_rejects_non_board_image() -> None:
    frame = np.full((841, 841, 3), (130, 180, 210), dtype=np.uint8)
    profile = BoardProfile(
        board_size=15,
        corners=((0, 0), (840, 0), (840, 840), (0, 840)),
    )

    result = recognize_frame(frame, profile)

    assert not result.board_visible


def test_low_contrast_blue_grid_uses_calibration_baseline() -> None:
    edge = 840
    frame = np.full((edge + 1, edge + 1, 3), (188, 178, 150), dtype=np.uint8)
    for coordinate in range(0, edge + 1, 60):
        cv2.line(frame, (coordinate, 0), (coordinate, edge), (195, 168, 138), 1)
        cv2.line(frame, (0, coordinate), (edge, coordinate), (195, 168, 138), 1)

    corners = ((0, 0), (edge, 0), (edge, edge), (0, edge))
    uncalibrated = BoardProfile(board_size=15, corners=corners)
    raw = recognize_frame(frame, uncalibrated)
    calibrated = BoardProfile(
        board_size=15,
        corners=corners,
        grid_score_baseline=raw.grid_score,
    )
    result = recognize_frame(frame, calibrated)

    assert 0.08 <= raw.grid_score < 0.35
    assert result.board_visible
    assert calibrated.grid_visibility_threshold < 0.35


def test_calibration_requires_a_valid_board_quadrilateral() -> None:
    assert is_valid_board_quadrilateral(
        ((100, 120), (700, 100), (720, 710), (80, 730)),
        (840, 840, 3),
    )
    assert not is_valid_board_quadrilateral(
        ((100, 100), (700, 100), (700, 110), (100, 110)),
        (840, 840, 3),
    )


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


def test_state_tracker_commits_the_first_black_stone_for_white_mode() -> None:
    board = BoardState.empty().set_cell(7, 7, Stone.BLACK)
    recognition = RecognitionResult(
        board=board,
        confidence=1.0,
        cell_confidences=(1.0,) * 225,
        board_visible=True,
        grid_score=1.0,
        warped=np.zeros((841, 841, 3), dtype=np.uint8),
    )
    tracker = StableStateTracker(required_frames=3)

    _, first = tracker.observe(recognition)
    _, second = tracker.observe(recognition)
    transition, third = tracker.observe(recognition)

    assert first is None
    assert second is None
    assert transition.valid
    assert transition.reason == "initial state"
    assert third == board


def test_state_tracker_pauses_for_banner_then_recovers() -> None:
    frame, profile = _synthetic_board()
    banner = frame.copy()
    cv2.rectangle(banner, (135, 580), (710, 665), (20, 20, 20), -1)
    tracker = StableStateTracker(required_frames=2)

    blocked, committed = tracker.observe(recognize_frame(banner, profile))
    _, first_clear = tracker.observe(recognize_frame(frame, profile))
    recovered, second_clear = tracker.observe(recognize_frame(frame, profile))

    assert not blocked.valid
    assert "covered" in blocked.reason
    assert committed is None
    assert first_clear is None
    assert recovered.valid
    assert second_clear is not None


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
