import json

from gomoku_assistant.benchmark import load_logged_positions


def _entry() -> dict[str, object]:
    board = [0] * 225
    board[7 * 15 + 7] = 1
    return {
        "board": board,
        "size": 15,
        "candidates": [{"rank": 1, "x": 7, "y": 6}],
    }


def test_benchmark_loads_legacy_and_schema_v2_sessions_once(tmp_path) -> None:
    (tmp_path / "legacy.json").write_text(json.dumps([_entry()]), encoding="utf-8")
    (tmp_path / "v2.json").write_text(
        json.dumps({"schema_version": 2, "analyses": [_entry()]}),
        encoding="utf-8",
    )

    positions = load_logged_positions(tmp_path)

    assert len(positions) == 1
    assert positions[0].baseline_move == (7, 6)
    assert positions[0].board.counts() == (1, 0)
