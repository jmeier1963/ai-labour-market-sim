"""Separations, vacancy posting, and unemployed-state updates."""

from labour_sim.config import SimConfig
from labour_sim.sim.labour_flows import post_vacancies, separations, update_unemployed
from labour_sim.sim.model import LabourMarketModel
from labour_sim.sim.workers import DISCOURAGED, EMPLOYED, SEARCHING


def make_model(seed: int = 11, n: int = 2000) -> LabourMarketModel:
    return LabourMarketModel(SimConfig(seed=seed, n_workers=n))


def test_separation_rate_close_to_config() -> None:
    model = make_model()
    employed_before = sum(1 for w in model.workers if w.state == EMPLOYED)
    separated = separations(model)
    rate = separated / employed_before
    assert 0.5 * model.cfg.labour.separation_rate < rate < 2.0 * model.cfg.labour.separation_rate


def test_vacancies_cover_headcount_deficit() -> None:
    model = make_model()
    separations(model)
    post_vacancies(model)
    for firm in model.firms:
        target = sum(firm.target_occupations.values())
        open_positions = firm.headcount + len(firm.vacancies)
        assert open_positions >= target, "firm should post vacancies for its full deficit"


def test_reservation_wage_decays_and_floors() -> None:
    model = make_model()
    worker = next(w for w in model.workers if w.state == SEARCHING)
    floor = model.cfg.policy.benefit_level * model.median_wage0
    worker.reservation_wage = floor * 1.5
    before = worker.reservation_wage
    update_unemployed(model)
    assert worker.reservation_wage < before
    for _ in range(600):
        update_unemployed(model)
    assert worker.reservation_wage >= floor * 0.999


def test_discouragement_after_long_spell() -> None:
    model = make_model()
    worker = next(w for w in model.workers if w.state == SEARCHING)
    worker.unemployment_ticks = model.cfg.matching.discouragement_ticks
    update_unemployed(model)
    assert worker.state == DISCOURAGED
