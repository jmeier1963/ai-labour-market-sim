"""Model skeleton: stepping preserves invariants; metrics collected each tick."""

from labour_sim.config import SimConfig
from labour_sim.sim.invariants import check_invariants
from labour_sim.sim.model import LabourMarketModel


def test_step_advances_clock_and_keeps_invariants() -> None:
    model = LabourMarketModel(SimConfig(seed=3, n_workers=1500))
    for _ in range(6):
        model.step()
        check_invariants(model)
    assert model.steps == 6


def test_invariants_across_seeds() -> None:
    for seed in range(5):
        model = LabourMarketModel(SimConfig(seed=seed, n_workers=1000))
        for _ in range(3):
            model.step()
        check_invariants(model)


def test_datacollector_records_each_tick() -> None:
    model = LabourMarketModel(SimConfig(seed=5, n_workers=1000))
    for _ in range(4):
        model.step()
    frame = model.datacollector.get_model_vars_dataframe()
    assert len(frame) == 4
    assert "unemployment_rate" in frame.columns
    assert frame["unemployment_rate"].between(0, 1).all()
