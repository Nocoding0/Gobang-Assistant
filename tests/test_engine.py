from gomoku_assistant.domain import BoardState, Stone
from gomoku_assistant.analysis import ProofStatus
from gomoku_assistant.engine import _build_rapfi_search_commands, parse_rapfi_output


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


def test_parse_rapfi_keeps_latest_algebraic_detail_for_final_move() -> None:
    board = BoardState.empty().set_cell(7, 7, Stone.BLACK)
    output = "\n".join(
        [
            "MESSAGE (1) 100 | 10-12 | H6",
            "MESSAGE (1) 200 | 12-18 | G7 F7",
            "6,6",
        ]
    )

    moves = parse_rapfi_output(output, board)

    assert len(moves) == 1
    assert (moves[0].x, moves[0].y) == (6, 6)
    assert moves[0].score == 200
    assert moves[0].principal_variation == ((6, 6), (5, 6))


def test_parse_rapfi_marks_forced_mate_from_algebraic_detail() -> None:
    board = BoardState.empty().set_cell(7, 7, Stone.BLACK)

    moves = parse_rapfi_output("MESSAGE (1) +M13 | 26-9 | H6\n7,5\n", board)

    assert len(moves) == 1
    assert (moves[0].x, moves[0].y) == (7, 5)
    assert moves[0].score is None
    assert moves[0].proof is ProofStatus.FORCED_WIN


def test_parse_rapfi_uses_final_summary_for_final_move_proof() -> None:
    board = BoardState.empty().set_cell(7, 7, Stone.BLACK)
    output = "\n".join(
        [
            "MESSAGE (1) 512 | 20-43 | I9",
            "MESSAGE Depth 20-43 | Eval +M43 | Time 2863ms | I9",
            "8,8",
        ]
    )

    moves = parse_rapfi_output(output, board)

    assert len(moves) == 1
    assert (moves[0].x, moves[0].y) == (8, 8)
    assert moves[0].score is None
    assert moves[0].proof is ProofStatus.FORCED_WIN


def test_rapfi_always_searches_one_principal_variation() -> None:
    board = (
        BoardState.empty()
        .set_cell(7, 7, Stone.BLACK)
        .set_cell(7, 6, Stone.WHITE)
    )

    commands = _build_rapfi_search_commands(board)

    assert "7,7,1" in commands
    assert "7,6,2" in commands
    assert commands[-2:] == ["DONE", "YXNBEST 1"]
