"""Search-and-matching market: applications, ranking, and hires.

Searchers send a handful of applications to vacancies they can plausibly fill,
weighted toward nearby occupations and acceptable offers. Firms rank applicants
by effective skill. The aggregate matching function is emergent.
"""

import math
from typing import TYPE_CHECKING

import numpy as np

from labour_sim.sim.wages import effective_skill, hire_wage
from labour_sim.sim.workers import SEARCHING, WorkerAgent

if TYPE_CHECKING:
    from labour_sim.sim.firms import Vacancy
    from labour_sim.sim.model import LabourMarketModel

# Short-spell searchers stay close to their occupation; long spells search anywhere.
NEAR_SEARCH_DISTANCE = 0.45
OFFER_TOLERANCE = 0.95  # apply if the (escalating) offer is within 5% of reservation


def _acceptable(
    model: "LabourMarketModel", worker: WorkerAgent, vacancy: "Vacancy", distance: float
) -> bool:
    max_distance = (
        1.0
        if worker.unemployment_ticks >= model.cfg.labour.retraining_wait_ticks
        else NEAR_SEARCH_DISTANCE
    )
    if distance > max_distance:
        return False
    return vacancy.wage_offer >= OFFER_TOLERANCE * worker.reservation_wage


def _collect_applications(
    model: "LabourMarketModel", searchers: list[WorkerAgent], vacancies: list["Vacancy"]
) -> None:
    rng = model.match_rng
    bias = model.cfg.matching.same_occupation_bias
    n_apps = model.cfg.matching.applications_per_searcher
    for worker in searchers:
        k = min(n_apps * 3, len(vacancies))  # oversample, keep the best-scoring few
        sampled = rng.choice(len(vacancies), size=k, replace=False)
        scored: list[tuple[float, "Vacancy"]] = []
        for idx in sampled:
            vacancy = vacancies[int(idx)]
            distance = model.occupation_distance(worker.occupation, vacancy.occupation)
            if not _acceptable(model, worker, vacancy, distance):
                continue
            scored.append((math.exp(bias * (1.0 - distance)), vacancy))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        for _, vacancy in scored[:n_apps]:
            vacancy.applicants.append(worker)


def run_matching(model: "LabourMarketModel") -> int:
    """One matching round; returns the number of hires."""
    vacancies = [v for firm in model.firms for v in firm.vacancies]
    searchers = [w for w in model.workers if w.state == SEARCHING]
    if not vacancies or not searchers:
        return 0

    order = model.match_rng.permutation(len(searchers))
    searchers = [searchers[int(i)] for i in order]
    _collect_applications(model, searchers, vacancies)

    hires = 0
    vacancy_order = model.match_rng.permutation(len(vacancies))
    for idx in vacancy_order:
        vacancy = vacancies[int(idx)]
        candidates = [w for w in vacancy.applicants if w.state == SEARCHING]
        if not candidates:
            continue
        # Screening/interviewing takes time: not every staffed vacancy closes this tick.
        if model.match_rng.uniform() > model.cfg.matching.efficiency:
            continue
        ranked = max(candidates, key=lambda w: effective_skill(model, w, vacancy.occupation))
        wage = hire_wage(model, ranked, vacancy.occupation, vacancy.firm)
        if wage < ranked.reservation_wage - 1e-12:
            continue
        if ranked.occupation != vacancy.occupation:
            key = (ranked.occupation, vacancy.occupation)
            model.occupation_flows[key] = model.occupation_flows.get(key, 0) + 1
            ranked.skill = effective_skill(model, ranked, vacancy.occupation)
            ranked.occupation = vacancy.occupation
        ranked.start_job(vacancy.firm, wage=wage)
        vacancy.firm.vacancies.remove(vacancy)
        hires += 1
    return hires


def market_tightness(model: "LabourMarketModel") -> float:
    vacancies = sum(len(f.vacancies) for f in model.firms)
    searchers = sum(1 for w in model.workers if w.state == SEARCHING)
    return vacancies / searchers if searchers else float(np.inf if vacancies else 0.0)
