"""Population and firm synthesis: composition matches the data, states are coherent."""

import numpy as np

from labour_sim.config import SimConfig
from labour_sim.sim.model import LabourMarketModel


def small_model(seed: int = 42) -> LabourMarketModel:
    return LabourMarketModel(SimConfig(seed=seed, n_workers=2000))


def test_worker_count_matches_config() -> None:
    model = small_model()
    assert len(model.workers) == 2000


def test_occupation_shares_approximate_data() -> None:
    model = small_model()
    counts: dict[str, int] = {}
    for w in model.workers:
        counts[w.occupation] = counts.get(w.occupation, 0) + 1
    for occ_id, occ in model.dataset.occupations.items():
        share = counts.get(occ_id, 0) / len(model.workers)
        assert abs(share - occ.employment_share) < 0.03, occ_id


def test_initial_unemployment_near_target() -> None:
    model = small_model()
    searching = sum(1 for w in model.workers if w.state == "searching")
    rate = searching / len(model.workers)
    assert 0.02 < rate < 0.10


def test_employment_links_consistent() -> None:
    model = small_model()
    for w in model.workers:
        if w.state == "employed":
            assert w.employer is not None
            assert w in w.employer.employees
        else:
            assert w.employer is None
    for f in model.firms:
        for w in f.employees:
            assert w.employer is f


def test_firm_sizes_positive_and_sector_mix_covered() -> None:
    model = small_model()
    assert all(len(f.employees) >= 1 for f in model.firms)
    sectors_present = {f.sector for f in model.firms}
    assert sectors_present == set(model.dataset.sectors)


def test_skills_lognormal_ish() -> None:
    model = small_model()
    skills = np.array([w.skill for w in model.workers])
    assert skills.min() > 0
    assert 0.9 < np.median(skills) < 1.1


def test_same_seed_same_population() -> None:
    a, b = small_model(7), small_model(7)
    for wa, wb in zip(a.workers, b.workers):
        assert wa.occupation == wb.occupation
        assert wa.skill == wb.skill
        assert wa.state == wb.state


def test_different_seed_different_population() -> None:
    a, b = small_model(1), small_model(2)
    skills_a = [w.skill for w in a.workers]
    skills_b = [w.skill for w in b.workers]
    assert skills_a != skills_b
