from gomoku_assistant.domain import (
    BoardState,
    Stone,
    infer_observed_moves,
    validate_transition,
)


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


def test_visual_transition_catches_up_two_legal_changes() -> None:
    previous = BoardState.empty()
    current = previous.set_cell(7, 7, Stone.BLACK).set_cell(8, 7, Stone.WHITE)

    result = validate_transition(previous, current)

    assert result.valid
    assert result.changed
    assert result.added_count == 2
    assert "caught up" in result.reason


def test_visual_transition_rejects_wrong_color() -> None:
    previous = BoardState.empty().place(7, 7, Stone.BLACK)
    current = previous.set_cell(8, 7, Stone.BLACK)

    result = validate_transition(previous, current)

    assert not result.valid
    assert "turn order" in result.reason


def test_visual_transition_rejects_bad_three_move_color_delta() -> None:
    previous = BoardState.empty().place(7, 7, Stone.BLACK)
    current = (
        previous.set_cell(8, 7, Stone.BLACK)
        .set_cell(9, 7, Stone.BLACK)
        .set_cell(10, 7, Stone.WHITE)
    )

    result = validate_transition(previous, current)

    assert not result.valid
    assert "turn order" in result.reason


def test_visual_transition_catches_up_three_legal_changes() -> None:
    previous = BoardState.empty().place(7, 7, Stone.BLACK)
    current = (
        previous.set_cell(8, 7, Stone.WHITE)
        .set_cell(9, 7, Stone.BLACK)
        .set_cell(10, 7, Stone.WHITE)
    )

    result = validate_transition(previous, current)

    assert result.valid
    assert result.added_count == 3


def test_observed_moves_number_a_single_placement() -> None:
    previous = BoardState.empty().place(7, 7, Stone.BLACK)
    current = previous.place(8, 7, Stone.WHITE)

    moves = infer_observed_moves(previous, current)

    assert [(move.x, move.y, move.number, move.certain) for move in moves] == [
        (8, 7, 2, True)
    ]


def test_observed_moves_number_two_move_catch_up_when_colors_are_unique() -> None:
    previous = BoardState.empty()
    current = previous.set_cell(7, 7, Stone.BLACK).set_cell(8, 7, Stone.WHITE)

    moves = infer_observed_moves(previous, current)

    assert {(move.x, move.y, move.number, move.certain) for move in moves} == {
        (7, 7, 1, True),
        (8, 7, 2, True),
    }


def test_observed_moves_do_not_invent_order_for_same_color_catch_up() -> None:
    previous = BoardState.empty()
    current = (
        previous.set_cell(7, 7, Stone.BLACK)
        .set_cell(8, 7, Stone.WHITE)
        .set_cell(9, 7, Stone.BLACK)
    )

    moves = infer_observed_moves(previous, current)

    assert {(move.x, move.number, move.certain) for move in moves} == {
        (7, None, False),
        (8, 2, True),
        (9, None, False),
    }
