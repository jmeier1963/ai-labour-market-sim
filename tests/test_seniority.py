"""Phase A seniority split: junior/senior variants of knowledge occupations.

Structure tests pin the data contract (pyramid shares, senior wage premium,
junior vectors more AI-exposed). Dynamics tests pin the emergent findings:
the junior:senior ratio stays stable without AI, inverts under fast takeoff,
and entrant/incumbent staff compositions CONVERGE (the real-world flat-entrant
contrast requires Phase B: promotion channel + organizational inertia).
"""

import math

import pytest

from labour_sim.config import SimConfig
from labour_sim.dataset import load_dataset
from labour_sim.sim.metrics import (
    knowledge_pyramid_ratio,
)
from labour_sim.sim.model import LabourMarketModel

SPLIT_BASES = ["SOC-13", "SOC-15", "SOC-17", "SOC-23", "SOC-27"]

FAST = {
    "capability": {"initial": 0.15, "ceiling": 0.98, "growth_rate": 0.08},
    "ai_cost": {"initial": 0.8, "decline_rate": 0.03},
    "adoption": {"imitation_weight": 0.6, "adjustment_cost": 0.15, "hurdle_mean": 0.1},
}
NO_AI = {"initial": 0.0, "ceiling": 0.0, "growth_rate": 0.0, "noise_sigma": 0.0}


def test_knowledge_occupations_split_into_pyramids() -> None:
    ds = load_dataset()
    for base in SPLIT_BASES:
        junior, senior = f"{base}-JR", f"{base}-SR"
        assert junior in ds.occupations, f"missing {junior}"
        assert senior in ds.occupations, f"missing {senior}"
        assert base not in ds.occupations, f"unsplit {base} still present"
        jr, sr = ds.occupations[junior], ds.occupations[senior]
        assert jr.employment_share > sr.employment_share, f"{base}: not a pyramid"
        assert sr.base_wage > jr.base_wage, f"{base}: no senior premium"


def test_junior_vectors_more_ai_exposed() -> None:
    ds = load_dataset()
    for base in SPLIT_BASES:
        def e2_mass(occ_id: str) -> float:
            occ = ds.occupations[occ_id]
            return sum(
                w for t, w in occ.task_weights.items() if ds.tasks[t].exposure == 2
            )

        assert e2_mass(f"{base}-JR") > e2_mass(f"{base}-SR") + 0.15, base


def test_junior_senior_distance_substantial() -> None:
    """The split should place junior and senior variants far enough apart that
    the ladder is not a free lateral move (Phase B adds the real channel)."""
    ds = load_dataset()
    for base in SPLIT_BASES:
        assert ds.occupation_distance(f"{base}-JR", f"{base}-SR") > 0.3, base


def test_pyramid_stable_without_ai() -> None:
    model = LabourMarketModel(SimConfig(seed=11, n_workers=2000, capability=NO_AI))
    model.run(24)
    start = knowledge_pyramid_ratio(model)
    model.run(216)
    end = knowledge_pyramid_ratio(model)
    assert not math.isnan(start) and not math.isnan(end)
    assert end > 0.7 * start, f"no-AI pyramid drifted: {start:.2f} -> {end:.2f}"


@pytest.fixture(scope="module")
def fast_model():
    model = LabourMarketModel(SimConfig(seed=42, n_workers=2000, **FAST))
    model.run()
    return model


def test_pyramid_inverts_under_fast_takeoff(fast_model) -> None:
    frame = fast_model.datacollector.get_model_vars_dataframe()
    start = frame["knowledge_pyramid_ratio"].iloc[:12].mean()
    end = frame["knowledge_pyramid_ratio"].iloc[-12:].mean()
    assert end < 0.6 * start, f"pyramid did not flatten: {start:.2f} -> {end:.2f}"


def test_entrant_and_incumbent_composition_converge(fast_model) -> None:
    """Documented finding, pinned: under Phase-A mechanics the inversion is
    near-universal — AI-native entrants and monthly-reoptimizing incumbents
    hold similar staff compositions through the cascade. The empirically
    observed flat-entrant/pyramidal-incumbent contrast therefore requires
    slower incumbent adjustment (organizational inertia) and/or an explicit
    apprenticeship pipeline: Phase B."""
    frame = fast_model.datacollector.get_model_vars_dataframe()
    window = frame.iloc[18:48]
    entrant = window["entrant_junior_share"].mean()
    incumbent = window["incumbent_junior_share"].mean()
    assert not math.isnan(entrant) and not math.isnan(incumbent)
    assert abs(entrant - incumbent) < 0.15, (
        f"compositions diverged: entrants {entrant:.2f} vs incumbents {incumbent:.2f} — "
        "if intended (e.g. Phase B), update this pin"
    )
