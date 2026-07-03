"""Firm labour demand: headcount targets from demand and adopted-task savings,
plus layoffs when targets fall below current staffing."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from labour_sim.sim.model import LabourMarketModel

MIN_PRODUCTIVITY_FACTOR = 0.05


def update_labour_demand(model: "LabourMarketModel") -> int:
    """Recompute per-occupation targets and lay off surplus workers.
    Returns the number of layoffs this tick."""
    firing_resistance = min(0.95, model.cfg.policy.firing_cost)
    layoffs = 0

    for firm in model.firms:
        firm.reset_tick_cache()
        targets: dict[str, int] = {}
        for occupation, base_count in firm.base_targets.items():
            auto, aug = firm.occupation_ai_effects(model, occupation)
            labour_per_output = max(MIN_PRODUCTIVITY_FACTOR, 1.0 - auto) / (1.0 + aug)
            targets[occupation] = round(base_count * firm.demand_factor * labour_per_output)
        firm.target_occupations = targets

        current: dict[str, list] = {}
        for worker in firm.employees:
            current.setdefault(worker.occupation, []).append(worker)
        for occupation, workers in current.items():
            surplus = len(workers) - targets.get(occupation, 0)
            if surplus <= 0:
                continue
            # Firing costs make firms shed surplus gradually rather than at once.
            workers_by_tenure = sorted(workers, key=lambda w: w.tenure)
            for worker in workers_by_tenure[:surplus]:
                if model.np_rng.uniform() < firing_resistance:
                    continue
                worker.lose_job()
                layoffs += 1
    return layoffs
