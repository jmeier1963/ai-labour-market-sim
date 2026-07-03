"""Firm adoption decisions: ROI logic, imitation, absorbing state."""

from labour_sim.config import SimConfig
from labour_sim.sim.adoption import adoption_probability, update_adoption
from labour_sim.sim.model import LabourMarketModel


def make_model(**config_overrides) -> LabourMarketModel:
    base = {"seed": 13, "n_workers": 1500}
    return LabourMarketModel(SimConfig(**{**base, **config_overrides}))


def test_zero_capability_means_zero_adoption() -> None:
    model = make_model(capability={"initial": 0.0, "ceiling": 0.0, "growth_rate": 0.0})
    for _ in range(24):
        model.step()
    assert all(not f.adopted for f in model.firms)


def test_cheap_capable_ai_gets_adopted() -> None:
    model = make_model(
        capability={"initial": 0.9, "ceiling": 0.95, "growth_rate": 0.0, "noise_sigma": 0.0},
        ai_cost={"initial": 0.05, "decline_rate": 0.0},
    )
    for _ in range(36):
        model.step()
    assert sum(1 for f in model.firms if f.adopted) > 0.5 * len(model.firms)


def test_adoption_probability_monotone_in_roi_and_peers() -> None:
    model = make_model()
    firm = next(iter(model.firms))
    low = adoption_probability(model, firm, roi=0.1, peer_share=0.0)
    high_roi = adoption_probability(model, firm, roi=0.6, peer_share=0.0)
    high_peers = adoption_probability(model, firm, roi=0.1, peer_share=0.8)
    assert high_roi > low
    assert high_peers > low


def test_adoption_is_absorbing() -> None:
    model = make_model(
        capability={"initial": 0.9, "ceiling": 0.95, "growth_rate": 0.0, "noise_sigma": 0.0},
        ai_cost={"initial": 0.05, "decline_rate": 0.0},
    )
    for _ in range(12):
        model.step()
    snapshot = {f.unique_id: set(f.adopted) for f in model.firms}
    update_adoption(model)
    for firm in model.firms:
        assert snapshot[firm.unique_id] <= firm.adopted, "adopted tasks must stay adopted"
