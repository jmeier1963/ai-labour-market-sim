"""Per-tick labour flows outside the matching market: separations, vacancy
posting, and unemployed-state updates (reservation decay, discouragement)."""

from typing import TYPE_CHECKING

from labour_sim.sim.firms import Vacancy
from labour_sim.sim.workers import DISCOURAGED, EMPLOYED, SEARCHING

if TYPE_CHECKING:
    from labour_sim.sim.model import LabourMarketModel

OFFER_ESCALATION = 0.02  # wage offer rises 2% per tick a vacancy stays unfilled
OFFER_CAP = 1.3


def separations(model: "LabourMarketModel") -> int:
    """Baseline quits/attrition: each employed worker separates w.p. separation_rate."""
    rate = model.cfg.labour.separation_rate
    separated = 0
    for worker in list(model.workers):
        if worker.state == EMPLOYED and model.np_rng.uniform() < rate:
            worker.lose_job()
            separated += 1
    return separated


def post_vacancies(model: "LabourMarketModel") -> None:
    """Firms post vacancies to close the gap to their occupation targets and
    escalate offers on positions that stay unfilled."""
    for firm in model.firms:
        current: dict[str, int] = {}
        for worker in firm.employees:
            current[worker.occupation] = current.get(worker.occupation, 0) + 1
        open_by_occ: dict[str, int] = {}
        for vacancy in firm.vacancies:
            vacancy.age += 1
            base = model.dataset.occupations[vacancy.occupation].base_wage
            vacancy.wage_offer = min(base * (1.0 + OFFER_ESCALATION * vacancy.age), base * OFFER_CAP)
            vacancy.applicants.clear()
            open_by_occ[vacancy.occupation] = open_by_occ.get(vacancy.occupation, 0) + 1

        for occupation, target in firm.target_occupations.items():
            deficit = target - current.get(occupation, 0) - open_by_occ.get(occupation, 0)
            base = model.dataset.occupations[occupation].base_wage
            for _ in range(max(0, deficit)):
                firm.vacancies.append(Vacancy(occupation=occupation, wage_offer=base, firm=firm))


def update_unemployed(model: "LabourMarketModel") -> None:
    """Reservation wages decay toward the benefit floor; long spells discourage."""
    cfg = model.cfg
    floor = cfg.policy.benefit_level * model.median_wage0
    for worker in model.workers:
        if worker.state == DISCOURAGED:
            # Discouraged workers drift back into search at a modest rate,
            # arriving with a floored reservation and full search breadth.
            if model.np_rng.uniform() < cfg.labour.reentry_rate:
                worker.state = SEARCHING
                worker.unemployment_ticks = cfg.labour.retraining_wait_ticks
                worker.reservation_wage = floor
            continue
        if worker.state != SEARCHING:
            continue
        worker.unemployment_ticks += 1
        decayed = worker.reservation_wage * (1.0 - cfg.matching.reservation_decay)
        worker.reservation_wage = max(decayed, floor)
        if worker.unemployment_ticks >= cfg.matching.discouragement_ticks:
            worker.state = DISCOURAGED
