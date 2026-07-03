"""Matching market: conservation, wage bounds, and edge cases."""

from labour_sim.config import SimConfig
from labour_sim.sim.invariants import check_invariants
from labour_sim.sim.labour_flows import post_vacancies, separations
from labour_sim.sim.matching import run_matching
from labour_sim.sim.model import LabourMarketModel
from labour_sim.sim.workers import EMPLOYED, SEARCHING


def prepared_model(seed: int = 21) -> LabourMarketModel:
    model = LabourMarketModel(SimConfig(seed=seed, n_workers=2000))
    separations(model)
    post_vacancies(model)
    return model


def test_hires_conserve_agents_and_slots() -> None:
    model = prepared_model()
    searchers_before = sum(1 for w in model.workers if w.state == SEARCHING)
    vacancies_before = sum(len(f.vacancies) for f in model.firms)
    hires = run_matching(model)
    searchers_after = sum(1 for w in model.workers if w.state == SEARCHING)
    vacancies_after = sum(len(f.vacancies) for f in model.firms)
    assert hires > 0, "a fresh market with vacancies must produce some hires"
    assert searchers_before - searchers_after == hires
    assert vacancies_before - vacancies_after == hires
    check_invariants(model)


def test_hired_wages_at_least_reservation() -> None:
    model = prepared_model()
    reservations = {id(w): w.reservation_wage for w in model.workers if w.state == SEARCHING}
    run_matching(model)
    for w in model.workers:
        if w.state == EMPLOYED and id(w) in reservations and w.tenure == 0 and w.employer:
            assert w.wage >= reservations[id(w)] - 1e-9


def test_no_hires_when_reservations_exceed_offers() -> None:
    model = prepared_model()
    for w in model.workers:
        if w.state == SEARCHING:
            w.reservation_wage = 1e9
    hires = run_matching(model)
    assert hires == 0


def test_no_vacancies_no_crash() -> None:
    model = LabourMarketModel(SimConfig(seed=5, n_workers=1000))
    for firm in model.firms:
        firm.vacancies.clear()
    assert run_matching(model) == 0


def test_matching_deterministic_per_seed() -> None:
    a, b = prepared_model(9), prepared_model(9)
    assert run_matching(a) == run_matching(b)
