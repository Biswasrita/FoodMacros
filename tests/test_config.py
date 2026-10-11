"""Stage 1 tests for configuration loading and validation."""

import json

import pytest

from foodmacros.config import (
    AppConfig,
    DEFAULT_CONFIG_PATH,
    config_from_dict,
    load_config,
)


def test_default_config_loads_and_matches_json():
    config = load_config(DEFAULT_CONFIG_PATH)
    assert isinstance(config, AppConfig)
    assert config.runtime.mode in {"quick", "full"}
    # The shipped default.json must round-trip to the same values.
    assert config.to_dict() == config_from_dict(
        json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    ).to_dict()


def test_missing_file_returns_defaults():
    config = load_config(DEFAULT_CONFIG_PATH.parent / "does_not_exist.json")
    assert config.to_dict() == AppConfig().to_dict()


def test_unknown_section_raises():
    with pytest.raises(ValueError, match="Unknown config section"):
        config_from_dict({"bogus_section": {}})


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown key"):
        config_from_dict({"runtime": {"not_a_real_key": 1}})


def test_partial_sections_fall_back_to_defaults():
    config = config_from_dict({"runtime": {"mode": "full"}})
    assert config.runtime.mode == "full"
    assert config.model.num_classes == AppConfig().model.num_classes


def test_save_and_reload_round_trip(tmp_path):
    config = AppConfig()
    config.runtime.mode = "full"
    path = config.save(tmp_path / "saved.json")
    reloaded = load_config(path)
    assert reloaded.to_dict() == config.to_dict()


def test_default_config_has_no_machine_specific_paths():
    raw = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    for forbidden in ("C:\\", "C:/", "/Users/", "/home/", "SANANDITA"):
        assert forbidden not in raw
