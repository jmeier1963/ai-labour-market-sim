"""Calibration gate: the no-AI baseline must hold realistic labour-market anchors.

These bands are the phase-3 contract. Any later mechanism (AI adoption,
entry/exit) must keep this baseline intact.
"""

import pytest

from labour_sim.config import SimConfig
from labour_sim.sim.invariants import check_invariants
from labour_sim.sim.model import LabourMarketModel

BURN_IN = 24


@pytest.fixture(scope="module")
def baseline_frame():
    no_ai = {"initial": 0.0, "ceiling": 0.0, "growth_rate": 0.0, "noise_sigma": 0.0}
    model = LabourMarketModel(SimConfig(seed=42, n_workers=2000, horizon_years=20, capability=no_ai))
    model.run()
    check_invariants(model)
    return model.datacollector.get_model_vars_dataframe().iloc[BURN_IN:]


def test_unemployment_in_band(baseline_frame) -> None:
    mean_u = baseline_frame["unemployment_rate"].mean()
    assert 0.03 < mean_u < 0.08, f"mean unemployment {mean_u:.3f} outside calibration band"
    assert baseline_frame["unemployment_rate"].max() < 0.12


def test_employment_stays_near_steady_state(baseline_frame) -> None:
    emp = baseline_frame["employment_rate"]
    assert abs(emp.iloc[-1] - emp.iloc[0]) < 0.03, "employment should not drift without AI"


def test_monthly_flow_rates_realistic(baseline_frame) -> None:
    finding = baseline_frame["job_finding_rate"].mean()
    separation = baseline_frame["separation_rate"].mean()
    assert 0.10 < finding < 0.45, f"job-finding {finding:.3f} outside band"
    assert 0.006 < separation < 0.025, f"separation {separation:.3f} outside band"


def test_wage_gini_plausible(baseline_frame) -> None:
    gini = baseline_frame["wage_gini"].iloc[-1]
    assert 0.20 < gini < 0.45, f"wage Gini {gini:.3f} outside plausible band"


def test_same_seed_reproduces_series() -> None:
    def run() -> str:
        model = LabourMarketModel(SimConfig(seed=7, n_workers=1000, horizon_years=5))
        model.run()
        frame = model.datacollector.get_model_vars_dataframe()
        return frame.to_csv()

    assert run() == run()
