"""Integration gates for the full AI dynamics (plan phases 4): emergence,
displacement-and-rebound, policy monotonicity, reproducibility."""

import numpy as np
import pytest

from labour_sim.config import SimConfig
from labour_sim.sim.invariants import check_invariants
from labour_sim.sim.model import LabourMarketModel

FAST = {
    "capability": {"initial": 0.15, "ceiling": 0.98, "growth_rate": 0.08},
    "ai_cost": {"initial": 0.8, "decline_rate": 0.03},
    "adoption": {"imitation_weight": 0.6, "adjustment_cost": 0.15, "hurdle_mean": 0.1},
}


@pytest.fixture(scope="module")
def fast_takeoff_frame():
    model = LabourMarketModel(SimConfig(seed=42, n_workers=2000, horizon_years=20, **FAST))
    model.run()
    check_invariants(model)
    return model.datacollector.get_model_vars_dataframe()


def test_perfect_cheap_ai_automates_and_disrupts(fast_takeoff_frame) -> None:
    frame = fast_takeoff_frame
    # Employed-weighted share; displaced workers in automated occupations drop
    # out of the base, so 0.25 of surviving task mass is already substantial.
    assert frame["automated_task_share"].iloc[-1] > 0.25
    assert frame["unemployment_rate"].max() > 0.08, "fast takeoff should spike unemployment"


def test_s_curve_emerges_not_instant(fast_takeoff_frame) -> None:
    adoption = fast_takeoff_frame["adoption_share"]
    final = adoption.iloc[-1]
    assert final > 0.4
    assert adoption.iloc[12] < 0.25 * final, "adoption must build up, not jump at t0"
    increments = adoption.diff().fillna(0)
    assert increments.iloc[24:].max() > increments.iloc[:12].max(), (
        "fastest adoption growth should come after an initial quiet phase (S-curve)"
    )


def test_output_expands_despite_displacement(fast_takeoff_frame) -> None:
    assert fast_takeoff_frame["output_index"].iloc[-1] > 1.05


def test_lower_matching_friction_weakly_lowers_peak_distress() -> None:
    """Reallocation friction is the one lever with a mechanically-signed effect:
    higher matching efficiency -> weakly lower peak distress.

    Policy levers are deliberately NOT gated: probing showed the model
    reproduces genuinely ambiguous trade-offs — benefit generosity raises
    reservation wages and slows re-employment; employment protection (firing
    costs) delays displacement into a synchronized, congested wave and can
    RAISE peak distress. Those are findings to analyse, not invariants."""

    def peak_distress(efficiency: float, seed: int) -> float:
        cfg = SimConfig(
            seed=seed,
            n_workers=1200,
            horizon_years=12,
            **FAST,
            matching={"efficiency": efficiency},
        )
        model = LabourMarketModel(cfg)
        model.run()
        frame = model.datacollector.get_model_vars_dataframe()
        return float((frame["unemployment_rate"] + frame["discouraged_share"]).max())

    seeds = [1, 2, 3, 4, 5]
    fluid = np.median([peak_distress(0.65, s) for s in seeds])
    frictional = np.median([peak_distress(0.25, s) for s in seeds])
    assert fluid <= frictional + 0.01, f"fluid {fluid:.3f} vs frictional {frictional:.3f}"


def test_ai_run_reproducible() -> None:
    def run() -> str:
        model = LabourMarketModel(SimConfig(seed=11, n_workers=1000, horizon_years=6, **FAST))
        model.run()
        return model.datacollector.get_model_vars_dataframe().to_csv()

    assert run() == run()
