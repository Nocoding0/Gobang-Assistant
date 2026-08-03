import json

from gomoku_assistant.analysis import (
    AnalysisResult,
    CandidateMove,
    ProofStatus,
    RecommendationMode,
    RejectedMove,
    SearchStats,
)
from gomoku_assistant.domain import BoardState, CorrectionEvent, Stone
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
        recommendation_mode=RecommendationMode.FORCED_DEFENSE,
        danger_points=((8, 7),),
        safe_candidate_count=1,
        rejected_moves=(
            RejectedMove(6, 7, "local", "allows an immediate opponent win", ((8, 7),)),
        ),
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
    assert payload["schema_version"] == 5
    assert payload["analyses"][0]["search"]["requested_time_ms"] == 15_000
    assert payload["analyses"][0]["recommendation_mode"] == "forced-defense"
    assert payload["analyses"][0]["danger_points"] == [[8, 7]]
    assert payload["analyses"][0]["rejected_moves"][0]["source"] == "local"


def test_session_records_manual_correction_events(tmp_path) -> None:
    logger = SessionLogger(tmp_path)

    logger.record_correction(
        CorrectionEvent(
            x=7,
            y=7,
            before=Stone.EMPTY,
            after=Stone.BLACK,
            vision=Stone.EMPTY,
            action="add",
        )
    )

    target = logger.save()

    assert target is not None
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["corrections"][0]["action"] == "add"
    assert payload["corrections"][0]["after"] == "black"
