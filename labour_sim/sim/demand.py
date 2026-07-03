"""Product market: automation lowers unit costs and prices; iso-elastic sector
demand expands in response (endogenous rebound); price competition shifts
market share between firms within a sector."""

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from labour_sim.sim.firms import FirmAgent
    from labour_sim.sim.model import LabourMarketModel

PASS_THROUGH = 0.6  # share of unit-cost savings passed into prices
PRICE_FLOOR = 0.3


def firm_cost_reduction(model: "LabourMarketModel", firm: "FirmAgent") -> float:
    """Unit-cost reduction in [0, ~1): automated task mass saves (1 - AI price),
    augmented (non-automated, exposed) task mass gets cheaper via productivity."""
    profile = firm.task_profile(model.dataset)
    if not profile:
        return 0.0
    saving = 0.0
    for task_id, mass in profile.items():
        task = model.dataset.tasks[task_id]
        quality = model.capability.task_quality(task)
        if task_id in firm.adopted and quality > 0:
            saving += mass * quality * max(0.0, 1.0 - model.ai_price)
        elif task.exposure > 0:
            boost = task.augmentation * model.capability.aug_level
            saving += mass * (boost / (1.0 + boost))
    return min(0.95, saving)


def update_product_market(model: "LabourMarketModel") -> None:
    """Set firm.price and firm.demand_factor; record sector demand indices."""
    trend = (1.0 + model.cfg.demand.growth_trend) ** model.steps
    sensitivity = model.cfg.demand.price_sensitivity

    by_sector: dict[str, list["FirmAgent"]] = {}
    for firm in model.firms:
        firm.price = max(PRICE_FLOOR, 1.0 - PASS_THROUGH * firm_cost_reduction(model, firm))
        by_sector.setdefault(firm.sector, []).append(firm)

    model.sector_demand = {}
    for sector_id, firms in by_sector.items():
        mean_price = sum(f.price for f in firms) / len(firms)
        elasticity = model.dataset.sectors[sector_id].demand_elasticity
        sector_index = trend * mean_price ** (-elasticity)
        model.sector_demand[sector_id] = sector_index

        # Size-weighted price competition over a fixed sector volume: total
        # volume = sector_index * t0 task capacity, split by size x price logit.
        # Equal prices at t0 give every firm demand_factor = 1; entrants dilute
        # incumbents proportionally instead of adding net demand.
        weights = [
            max(1, sum(f.base_targets.values()))
            * math.exp(-sensitivity * (f.price - mean_price))
            for f in firms
        ]
        total_weight = sum(weights)
        sector_volume = sector_index * model.sector_size_base[sector_id]
        for firm, weight in zip(firms, weights):
            base_size = max(1, sum(firm.base_targets.values()))
            firm.demand_factor = sector_volume * (weight / total_weight) / base_size
