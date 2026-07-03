"""Monte Carlo batches, percentile bands, parameter sweeps, scenario files."""

import pytest

from labour_sim.batch import monte_carlo, set_by_path, summary_stat, sweep_1d
from labour_sim.config import SimConfig
from labour_sim.scenarios import Scenario, scenario_from_json, scenario_to_json

SMALL = {"n_workers": 800, "horizon_years": 3}


def test_monte_carlo_runs_and_bands() -> None:
    result = monte_carlo(SimConfig(seed=5, **SMALL), runs=4, processes=1)
    assert len(result.runs) == 4
    bands = result.bands("unemployment_rate")
    assert list(bands.columns) == ["p5", "p25", "p50", "p75", "p95"]
    assert (bands["p5"] <= bands["p50"]).all()
    assert (bands["p50"] <= bands["p95"]).all()
    assert len(bands) == 36  # 3 years of monthly ticks


def test_monte_carlo_reproducible() -> None:
    a = monte_carlo(SimConfig(seed=9, **SMALL), runs=3, processes=1)
    b = monte_carlo(SimConfig(seed=9, **SMALL), runs=3, processes=1)
    assert a.bands("unemployment_rate").equals(b.bands("unemployment_rate"))


def test_runs_differ_across_sub_seeds() -> None:
    result = monte_carlo(SimConfig(seed=5, **SMALL), runs=3, processes=1)
    u_finals = {frame["unemployment_rate"].iloc[-1] for frame in result.runs}
    assert len(u_finals) > 1, "different sub-seeds must yield different paths"


def test_set_by_path() -> None:
    cfg = set_by_path(SimConfig(), "capability.growth_rate", 0.09)
    assert cfg.capability.growth_rate == 0.09
    with pytest.raises(KeyError):
        set_by_path(SimConfig(), "capability.nonexistent", 1.0)


def test_summary_stats() -> None:
    result = monte_carlo(SimConfig(seed=5, **SMALL), runs=2, processes=1)
    frame = result.runs[0]
    assert 0.0 <= summary_stat(frame, "peak_unemployment") <= 1.0
    assert summary_stat(frame, "final_output") > 0.0


def test_sweep_1d_grid() -> None:
    table = sweep_1d(
        SimConfig(seed=3, **SMALL),
        "capability.growth_rate",
        [0.0, 0.08],
        stat="peak_unemployment",
        runs_per_point=2,
        processes=1,
    )
    assert list(table["value"]) == [0.0, 0.08]
    assert (table["stat_median"] >= 0).all()


def test_scenario_roundtrip() -> None:
    scenario = Scenario(name="test", notes="hello", config=SimConfig(seed=123, n_workers=900))
    restored = scenario_from_json(scenario_to_json(scenario))
    assert restored.config == scenario.config
    assert restored.name == "test"


def test_scenario_rejects_bad_payload() -> None:
    with pytest.raises(ValueError):
        scenario_from_json('{"name": "x", "config": {"n_workers": -1}}')
