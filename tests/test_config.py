"""SimConfig validation and preset loading."""

import pytest
from pydantic import ValidationError

from labour_sim.config import SimConfig, load_preset, list_presets


def test_default_config_is_valid() -> None:
    cfg = SimConfig()
    assert cfg.seed == 42
    assert cfg.n_workers >= 1000
    assert cfg.ticks == cfg.horizon_years * 12


def test_invalid_values_rejected() -> None:
    with pytest.raises(ValidationError):
        SimConfig(n_workers=-5)
    with pytest.raises(ValidationError):
        SimConfig(capability={"growth_rate": -1.0})
    with pytest.raises(ValidationError):
        SimConfig(matching={"bargaining_beta": 1.5})


def test_presets_load_and_differ_from_baseline() -> None:
    names = list_presets()
    assert {"baseline", "fast_takeoff", "high_friction", "policy_cushion"} <= set(names)
    baseline = load_preset("baseline")
    fast = load_preset("fast_takeoff")
    assert isinstance(baseline, SimConfig)
    assert fast.capability.growth_rate > baseline.capability.growth_rate


def test_unknown_preset_raises() -> None:
    with pytest.raises(KeyError):
        load_preset("does_not_exist")


def test_config_roundtrips_through_json() -> None:
    cfg = SimConfig(seed=7, n_workers=2000)
    restored = SimConfig.model_validate_json(cfg.model_dump_json())
    assert restored == cfg
