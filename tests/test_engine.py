from gomoku_assistant.domain import BoardState, Stone
from gomoku_assistant.engine import parse_rapfi_output


def test_parse_rapfi_detail_lines_and_final_move() -> None:
    board = BoardState.empty().set_cell(7, 7, Stone.BLACK)
    output = "\n".join(
        [
            "MESSAGE (1) 321 | 12-16 | 8,7 8,8 9,7",
            "MESSAGE (2) 280 | 12-15 | 6,7 7,8",
            "MESSAGE (3) -30 | 11-14 | 7,6 6,6",
            "8,7",
        ]
    )

    moves = parse_rapfi_output(output, board)

    assert [(move.x, move.y) for move in moves] == [(8, 7), (6, 7), (7, 6)]
    assert [move.rank for move in moves] == [1, 2, 3]
    assert moves[0].score == 321


def test_parse_rapfi_uses_final_move_when_detail_is_missing() -> None:
    board = BoardState.empty()

    moves = parse_rapfi_output("MESSAGE ready\n7,7\n", board)

    assert len(moves) == 1
    assert (moves[0].x, moves[0].y) == (7, 7)

