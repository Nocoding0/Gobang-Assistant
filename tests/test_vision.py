import cv2
import numpy as np

from gomoku_assistant.domain import Stone
from gomoku_assistant.vision import BoardProfile, StableStateTracker, recognize_frame


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

