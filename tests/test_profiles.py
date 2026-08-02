import json

from gomoku_assistant.capture import WindowInfo
from gomoku_assistant.profiles import ProfileStore
from gomoku_assistant.vision import BoardProfile


def test_profile_store_uses_window_title_and_size(tmp_path) -> None:
    store = ProfileStore(tmp_path)
    profile = BoardProfile(
        board_size=15,
        corners=((0, 0), (840, 0), (840, 840), (0, 840)),
        source_width=522,
        source_height=981,
        window_title="Gomoku",
    )
    store.save(profile)

    matching = WindowInfo(1, "Gomoku", 0, 0, 522, 981)
    different_size = WindowInfo(2, "Gomoku", 0, 0, 412, 776)

    loaded = store.load_for(matching)

    assert loaded == profile
    assert store.load_for(different_size) is None


def test_profile_store_migrates_matching_legacy_default(tmp_path) -> None:
    legacy = BoardProfile(
        board_size=15,
        corners=((0, 0), (840, 0), (840, 840), (0, 840)),
        schema_version=1,
        source_width=412,
        source_height=776,
        window_title="Legacy Gomoku",
    )
    (tmp_path / "default.json").write_text(
        json.dumps(legacy.to_dict()),
        encoding="utf-8",
    )
    store = ProfileStore(tmp_path)
    window = WindowInfo(1, "Legacy Gomoku", 0, 0, 412, 776)

    loaded = store.load_for(window)

    assert loaded == legacy
    assert len(list(tmp_path.glob("profile-*.json"))) == 1


def test_profile_preserves_grid_score_baseline() -> None:
    profile = BoardProfile(
        board_size=15,
        corners=((0, 0), (840, 0), (840, 840), (0, 840)),
        grid_score_baseline=0.12,
        black_disk_fraction_min=0.61,
        white_disk_fraction_min=0.57,
        black_saturation_max=71.0,
        black_low_saturation_fraction_min=0.63,
    )

    restored = BoardProfile.from_dict(profile.to_dict())

    assert restored == profile
    assert restored.grid_visibility_threshold == 0.08
