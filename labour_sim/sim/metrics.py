"""Model-level metric reporters for the Mesa DataCollector."""

from typing import TYPE_CHECKING, Callable

import numpy as np

from labour_sim.sim.workers import DISCOURAGED, EMPLOYED, SEARCHING

if TYPE_CHECKING:
    from labour_sim.sim.model import LabourMarketModel


def _states(model: "LabourMarketModel") -> tuple[int, int, int]:
    employed = searching = discouraged = 0
    for worker in model.workers:
        if worker.state == EMPLOYED:
            employed += 1
        elif worker.state == SEARCHING:
            searching += 1
        elif worker.state == DISCOURAGED:
            discouraged += 1
    return employed, searching, discouraged


def unemployment_rate(model: "LabourMarketModel") -> float:
    employed, searching, _ = _states(model)
    labour_force = employed + searching
    return searching / labour_force if labour_force else 0.0


def employment_rate(model: "LabourMarketModel") -> float:
    employed, _, _ = _states(model)
    return employed / len(model.workers)


def mean_wage(model: "LabourMarketModel") -> float:
    wages = [w.wage for w in model.workers if w.state == EMPLOYED]
    return float(np.mean(wages)) if wages else 0.0


def wage_gini(model: "LabourMarketModel") -> float:
    wages = np.sort(np.array([w.wage for w in model.workers if w.state == EMPLOYED]))
    if wages.size == 0 or wages.sum() == 0:
        return 0.0
    n = wages.size
    index = np.arange(1, n + 1)
    return float((2 * (index * wages).sum() - (n + 1) * wages.sum()) / (n * wages.sum()))


def n_firms(model: "LabourMarketModel") -> int:
    return len(model.firms)


def job_finding_rate(model: "LabourMarketModel") -> float:
    return model.flow_hires / model.flow_searchers if model.flow_searchers else 0.0


def separation_rate(model: "LabourMarketModel") -> float:
    if not model.flow_employed_start:
        return 0.0
    return model.flow_separations / model.flow_employed_start


def discouraged_share(model: "LabourMarketModel") -> float:
    _, _, discouraged = _states(model)
    return discouraged / len(model.workers)


def vacancy_count(model: "LabourMarketModel") -> int:
    return sum(len(f.vacancies) for f in model.firms)


def capability_level(model: "LabourMarketModel") -> float:
    return model.capability.level


def ai_price(model: "LabourMarketModel") -> float:
    return model.ai_price


def adoption_share(model: "LabourMarketModel") -> float:
    """Mean share of firms' task mass performed by AI (unweighted firm mean)."""
    firms = list(model.firms)
    if not firms:
        return 0.0
    return sum(f.adoption_share() for f in firms) / len(firms)


def automated_task_share(model: "LabourMarketModel") -> float:
    """Quality-weighted share of employed workers' task mass automated by their employer."""
    total = 0.0
    automated = 0.0
    for worker in model.workers:
        if worker.state != EMPLOYED or worker.employer is None:
            continue
        weights = model.dataset.occupations[worker.occupation].task_weights
        for task_id, weight in weights.items():
            total += weight
            if task_id in worker.employer.adopted:
                quality = model.capability.task_quality(model.dataset.tasks[task_id])
                automated += weight * quality
    return automated / total if total else 0.0


def output_index(model: "LabourMarketModel") -> float:
    """Economy output volume relative to t0 (sector demand indices weighted by
    t0 sector task capacity)."""
    total_base = sum(model.sector_size_base.values())
    if not total_base or not model.sector_demand:
        return 1.0
    volume = sum(
        model.sector_demand.get(sector_id, 1.0) * base
        for sector_id, base in model.sector_size_base.items()
    )
    return volume / total_base


def layoff_rate(model: "LabourMarketModel") -> float:
    if not model.flow_employed_start:
        return 0.0
    return model.flow_layoffs / model.flow_employed_start


def _wage_percentile(percentile: int):
    def reporter(model: "LabourMarketModel") -> float:
        wages = [w.wage for w in model.workers if w.state == EMPLOYED]
        return float(np.percentile(wages, percentile)) if wages else 0.0

    reporter.__name__ = f"wage_p{percentile}"
    return reporter


def market_tightness_metric(model: "LabourMarketModel") -> float:
    searchers = sum(1 for w in model.workers if w.state == SEARCHING)
    return vacancy_count(model) / searchers if searchers else 0.0


def model_reporters() -> dict[str, Callable]:
    return {
        "unemployment_rate": unemployment_rate,
        "employment_rate": employment_rate,
        "mean_wage": mean_wage,
        "wage_gini": wage_gini,
        "n_firms": n_firms,
        "job_finding_rate": job_finding_rate,
        "separation_rate": separation_rate,
        "discouraged_share": discouraged_share,
        "vacancy_count": vacancy_count,
        "capability_level": capability_level,
        "ai_price": ai_price,
        "adoption_share": adoption_share,
        "automated_task_share": automated_task_share,
        "output_index": output_index,
        "layoff_rate": layoff_rate,
        "market_tightness": market_tightness_metric,
        **{f"wage_p{p}": _wage_percentile(p) for p in (10, 25, 50, 75, 90)},
    }
