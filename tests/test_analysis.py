from gomoku_assistant.analysis import (
    CandidateMove,
    HeuristicAnalyzer,
    ProofStatus,
    RecommendationMode,
    assess_tactical_position,
    filter_safe_candidates,
    immediate_winning_points,
    validate_candidate_safety,
)
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
    assert result.recommendation_mode is RecommendationMode.WIN_NOW
    assert len(result.candidates) == 2


def test_two_opponent_wins_are_reported_as_no_single_move_defense() -> None:
    board = BoardState.empty()
    for x in range(4, 8):
        board = board.set_cell(x, 7, Stone.WHITE)
    board = board.set_cell(0, 0, Stone.BLACK)
    board = board.set_cell(2, 2, Stone.BLACK)
    board = board.set_cell(4, 4, Stone.BLACK)
    board = board.set_cell(6, 6, Stone.BLACK)

    result = HeuristicAnalyzer().analyze(board)

    assert result.recommendation_mode is RecommendationMode.FORCED_LOSS
    assert result.candidates == ()
    assert set(result.danger_points) == {(3, 7), (8, 7)}


def test_screenshot_regression_shows_only_l11_as_forced_defense() -> None:
    board = BoardState.empty()
    # Black has already occupied G6, so White's H7-I8-J9-K10 diagonal can
    # only be completed at L11. Black is to move (eight stones each).
    for x, y in ((0, 0), (6, 5), (7, 5), (5, 6), (9, 6), (6, 7), (7, 7), (8, 8)):
        board = board.set_cell(x, y, Stone.BLACK)
    for x, y in ((12, 0), (6, 6), (7, 6), (8, 6), (10, 6), (8, 7), (9, 8), (10, 9)):
        board = board.set_cell(x, y, Stone.WHITE)

    assert immediate_winning_points(board, Stone.WHITE) == ((11, 10),)
    result = HeuristicAnalyzer().analyze(board)

    assert result.recommendation_mode is RecommendationMode.FORCED_DEFENSE
    assert [(move.x, move.y) for move in result.candidates] == [(11, 10)]
    assert (8, 5) not in [(move.x, move.y) for move in result.candidates]
    assert (5, 5) not in [(move.x, move.y) for move in result.candidates]


def test_filter_does_not_pad_two_safe_input_moves() -> None:
    board = BoardState.empty().set_cell(7, 7, Stone.BLACK)
    ranked = (
        ("test", CandidateMove(7, 6, 1, 1, ProofStatus.HEURISTIC)),
        ("test", CandidateMove(6, 7, 2, 1, ProofStatus.HEURISTIC)),
    )

    candidates, rejected = filter_safe_candidates(board, ranked, limit=3)

    assert len(candidates) == 2
    assert rejected == ()


def test_rejects_forced_win_claim_that_allows_immediate_loss() -> None:
    board = BoardState.empty()
    for x in range(4, 8):
        board = board.set_cell(x, 7, Stone.WHITE)
    for x, y in ((0, 0), (1, 1), (2, 2), (3, 3)):
        board = board.set_cell(x, y, Stone.BLACK)
    unsafe = CandidateMove(10, 10, 1, None, ProofStatus.FORCED_WIN)

    candidates, rejected = filter_safe_candidates(board, (("rapfi", unsafe),), limit=3)

    assert candidates == ()
    assert rejected[0].source == "rapfi"
    assert rejected[0].reason == "allows an immediate opponent win"
    assert set(rejected[0].danger_points) == {(3, 7), (8, 7)}


def test_rejects_move_that_permits_an_unanswered_double_threat() -> None:
    board = BoardState.empty()
    for x, y in ((0, 0), (1, 0), (2, 0)):
        board = board.set_cell(x, y, Stone.BLACK)
    for x, y in ((6, 7), (7, 7), (9, 7)):
        board = board.set_cell(x, y, Stone.WHITE)

    safety = validate_candidate_safety(board, (1, 1))

    assert safety is not None
    reason, danger_points = safety
    assert reason == "allows an opponent double threat"
    assert set(danger_points) == {(5, 7), (10, 7)}
