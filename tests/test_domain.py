from gomoku_assistant.domain import BoardState, Stone, validate_transition


def test_empty_board_starts_with_black() -> None:
    board = BoardState.empty()
    assert board.side_to_move() is Stone.BLACK
    assert board.is_count_legal()


def test_freestyle_overline_is_a_win() -> None:
    board = BoardState.empty()
    for x in range(6):
        board = board.set_cell(x, 7, Stone.BLACK)

    line = board.winning_line()
    assert line is not None
    assert line.stone is Stone.BLACK
    assert len(line.points) == 6


def test_legal_visual_transition_adds_expected_color() -> None:
    previous = BoardState.empty().place(7, 7, Stone.BLACK)
    current = previous.place(8, 7, Stone.WHITE)

    result = validate_transition(previous, current)

    assert result.valid
    assert result.changed


def test_visual_transition_rejects_two_changes() -> None:
    previous = BoardState.empty()
    current = previous.set_cell(7, 7, Stone.BLACK).set_cell(8, 7, Stone.WHITE)

    result = validate_transition(previous, current)

    assert not result.valid
    assert "more than one" in result.reason


def test_visual_transition_rejects_wrong_color() -> None:
    previous = BoardState.empty().place(7, 7, Stone.BLACK)
    current = previous.set_cell(8, 7, Stone.BLACK)

    result = validate_transition(previous, current)

    assert not result.valid
    assert "wrong color" in result.reason

