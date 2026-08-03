import json

from gomoku_assistant.analysis import AnalysisResult, CandidateMove, ProofStatus, SearchStats
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
        search_stats=SearchStats(
            requested_time_ms=15_000,
            elapsed_ms=14_822,
            engine_time_ms=14_799,
            depth="22-36",
            threads=8,
            hash_kib=512 * 1024,
        ),
    )
    logger = SessionLogger(tmp_path)

    logger.append(result)

    assert logger.entries[0].board == tuple(int(cell) for cell in engine_board.cells)
    assert logger.entries[0].size == 15
    assert logger.entries[0].search is not None
    assert logger.entries[0].search["depth"] == "22-36"

    target = logger.save()

    assert target is not None
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["analyses"][0]["search"]["requested_time_ms"] == 15_000
