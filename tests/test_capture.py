import numpy as np

from gomoku_assistant.capture import is_blank_frame
from gomoku_assistant.vision import BoardProfile


def test_black_frame_is_rejected() -> None:
    assert is_blank_frame(np.zeros((100, 200, 3), dtype=np.uint8))


def test_nearly_black_frame_is_rejected() -> None:
    frame = np.full((100, 200, 3), 4, dtype=np.uint8)
    assert is_blank_frame(frame)


def test_visible_frame_is_not_rejected() -> None:
    frame = np.full((100, 200, 3), (95, 170, 220), dtype=np.uint8)
    assert not is_blank_frame(frame)


def test_profile_requires_matching_capture_source() -> None:
    profile = BoardProfile(
        board_size=15,
        corners=((0, 0), (840, 0), (840, 840), (0, 840)),
        source_width=428,
        source_height=784,
        window_title="Two Player Gomoku",
    )
    frame = np.zeros((784, 428, 3), dtype=np.uint8)

    assert profile.matches_source(frame, "Two Player Gomoku")
    assert not profile.matches_source(frame, "Different window")
    assert not profile.matches_source(np.zeros((800, 428, 3), dtype=np.uint8), "Two Player Gomoku")
