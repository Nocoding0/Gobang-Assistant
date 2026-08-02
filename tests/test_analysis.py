from gomoku_assistant.analysis import HeuristicAnalyzer, ProofStatus
from gomoku_assistant.domain import BoardState, Stone


def test_heuristic_takes_immediate_win() -> None:
    board = BoardState.empty()
    for x in range(4, 8):
        board = board.set_cell(x, 7, Stone.BLACK)
    board = board.set_cell(2, 2, Stone.WHITE)
    board = board.set_cell(3, 3, Stone.WHITE)
    board = board.set_cell(4, 4, Stone.WHITE)
    board = board.set_cell(5, 5, Stone.WHITE)

    result = HeuristicAnalyzer().analyze(board)

    assert result.candidates
    assert result.candidates[0].proof is ProofStatus.WIN_IN_ONE
    assert (result.candidates[0].x, result.candidates[0].y) in {(3, 7), (8, 7)}


def test_heuristic_blocks_opponent_win() -> None:
    board = BoardState.empty()
    for x in range(4, 8):
        board = board.set_cell(x, 7, Stone.WHITE)
    board = board.set_cell(0, 0, Stone.BLACK)
    board = board.set_cell(2, 2, Stone.BLACK)
    board = board.set_cell(4, 4, Stone.BLACK)
    board = board.set_cell(6, 6, Stone.BLACK)

    result = HeuristicAnalyzer().analyze(board)

    assert result.candidates
    assert (result.candidates[0].x, result.candidates[0].y) in {(3, 7), (8, 7)}
