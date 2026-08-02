from gomoku_assistant.analysis import AnalysisResult, CandidateMove, ProofStatus
from gomoku_assistant.domain import BoardState, Stone
from gomoku_assistant.sessions import SessionLogger


def test_session_records_the_board_analyzed_by_the_engine(tmp_path) -> None:
    engine_board = BoardState.empty().set_cell(7, 7, Stone.BLACK)
    result = AnalysisResult(
        board=engine_board,
        candidates=(
            CandidateMove(
                x=7,
                y=6,
                rank=1,
                score=123,
                proof=ProofStatus.HEURISTIC,
            ),
        ),
        engine_name="Rapfi",
    )
    logger = SessionLogger(tmp_path)

    logger.append(result)

    assert logger.entries[0].board == tuple(int(cell) for cell in engine_board.cells)
    assert logger.entries[0].size == 15
