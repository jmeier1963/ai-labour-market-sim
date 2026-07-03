"""Initial population and firm synthesis from the bundled dataset.

Occupation counts use largest-remainder apportionment so realized shares track
the data tightly even for small populations; workers are then allocated to
sectors in proportion to each sector's demand for their occupation, and packed
into firms with truncated-Pareto sizes.
"""

from typing import TYPE_CHECKING

import numpy as np

from labour_sim.sim.firms import FirmAgent
from labour_sim.sim.workers import SEARCHING, WorkerAgent

if TYPE_CHECKING:
    from labour_sim.sim.model import LabourMarketModel

FIRM_SIZE_CAP = 400
SKILL_SIGMA = 0.35


def _apportion(total: int, shares: dict[str, float]) -> dict[str, int]:
    """Largest-remainder apportionment of `total` slots over normalized shares."""
    quotas = {k: total * v for k, v in shares.items()}
    counts = {k: int(q) for k, q in quotas.items()}
    remainder = total - sum(counts.values())
    by_frac = sorted(quotas, key=lambda k: quotas[k] - counts[k], reverse=True)
    for key in by_frac[:remainder]:
        counts[key] += 1
    return counts


def synthesize(model: "LabourMarketModel") -> None:
    cfg, dataset, rng = model.cfg, model.dataset, model.np_rng

    occ_counts = _apportion(
        cfg.n_workers, {o: d.employment_share for o, d in dataset.occupations.items()}
    )

    # Allocate each occupation's workers across sectors by sector demand for it.
    sector_workers: dict[str, list[WorkerAgent]] = {s: [] for s in dataset.sectors}
    for occ_id, count in occ_counts.items():
        if count == 0:
            continue
        demand = {
            s: d.employment_share * d.occupation_mix.get(occ_id, 0.0)
            for s, d in dataset.sectors.items()
        }
        total_demand = sum(demand.values())
        if total_demand <= 0:
            raise ValueError(f"no sector demands occupation {occ_id}; fix sectors.json")
        allocation = _apportion(count, {s: v / total_demand for s, v in demand.items()})
        skills = np.exp(rng.normal(0.0, SKILL_SIGMA, size=count))
        i = 0
        for sector_id, n in allocation.items():
            for _ in range(n):
                worker = WorkerAgent(model, occupation=occ_id, skill=float(skills[i]))
                sector_workers[sector_id].append(worker)
                i += 1

    for sector_id, workers in sector_workers.items():
        _build_firms(model, sector_id, workers)

    for firm in model.firms:
        targets: dict[str, int] = {}
        for worker in firm.employees:
            targets[worker.occupation] = targets.get(worker.occupation, 0) + 1
        firm.target_occupations = targets
        firm.base_targets = dict(targets)

    wages = sorted(w.wage for w in model.workers)
    model.median_wage0 = float(wages[len(wages) // 2])

    _mark_initial_unemployed(model)


def _draw_firm_sizes(model: "LabourMarketModel", sector_id: str, n_workers: int) -> list[int]:
    sector = model.dataset.sectors[sector_id]
    sizes: list[int] = []
    remaining = n_workers
    while remaining > 0:
        u = model.np_rng.uniform()
        size = int(sector.firm_size_min * (1.0 - u) ** (-1.0 / sector.firm_size_alpha))
        size = max(sector.firm_size_min, min(size, FIRM_SIZE_CAP, remaining))
        # Avoid a dangling sub-minimum firm at the end.
        if 0 < remaining - size < sector.firm_size_min:
            size = remaining
        sizes.append(size)
        remaining -= size
    return sizes


def _build_firms(model: "LabourMarketModel", sector_id: str, workers: list[WorkerAgent]) -> None:
    cfg, rng = model.cfg, model.np_rng
    pool = list(workers)
    rng.shuffle(pool)  # type: ignore[arg-type]
    cursor = 0
    for size in _draw_firm_sizes(model, sector_id, len(pool)):
        hurdle = max(0.0, rng.normal(cfg.adoption.hurdle_mean, cfg.adoption.hurdle_sigma))
        firm = FirmAgent(model, sector=sector_id, adoption_hurdle=float(hurdle))
        for worker in pool[cursor : cursor + size]:
            wage = model.dataset.occupations[worker.occupation].base_wage * worker.skill
            worker.start_job(firm, wage=float(wage))
        cursor += size


def _mark_initial_unemployed(model: "LabourMarketModel") -> None:
    workers = list(model.workers)
    n_unemployed = round(model.cfg.labour.initial_unemployment * len(workers))
    indices = model.np_rng.choice(len(workers), size=n_unemployed, replace=False)
    for i in indices:
        worker = workers[int(i)]
        wage = worker.wage
        worker.lose_job()
        worker.state = SEARCHING
        worker.reservation_wage = 0.9 * wage
