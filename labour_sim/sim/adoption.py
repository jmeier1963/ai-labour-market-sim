"""Firm AI-adoption decisions: sampled-task ROI evaluation with imitation.

No sigmoid over time anywhere: S-curves emerge from capability crossing task
difficulties, falling AI prices, heterogeneous hurdles, and peer imitation.
"""

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from labour_sim.sim.firms import FirmAgent
    from labour_sim.sim.model import LabourMarketModel


def adoption_probability(
    model: "LabourMarketModel", firm: "FirmAgent", roi: float, peer_share: float
) -> float:
    cfg = model.cfg.adoption
    signal = roi + cfg.imitation_weight * peer_share - firm.adoption_hurdle
    x = cfg.logistic_slope * signal
    if x < -60.0:
        return 0.0
    if x > 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def sector_adoption_shares(model: "LabourMarketModel") -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for firm in model.firms:
        totals.setdefault(firm.sector, []).append(firm.adoption_share())
    return {s: sum(v) / len(v) for s, v in totals.items()}


def _mean_wage(firm: "FirmAgent", model: "LabourMarketModel") -> float:
    if firm.employees:
        return firm.wage_bill() / firm.headcount
    # Entrants without staff price labour at their target occupations' base wages.
    total = sum(firm.target_occupations.values())
    if not total:
        return model.median_wage0
    return (
        sum(
            model.dataset.occupations[occ].base_wage * count
            for occ, count in firm.target_occupations.items()
        )
        / total
    )


def update_adoption(model: "LabourMarketModel") -> None:
    """Each firm re-evaluates a few tasks from its task profile per tick."""
    cfg = model.cfg.adoption
    peer_shares = sector_adoption_shares(model)
    ai_price = model.ai_price

    for firm in model.firms:
        profile = firm.task_profile(model.dataset)
        candidates = [t for t in profile if t not in firm.adopted]
        if not candidates:
            continue
        weights = [profile[t] for t in candidates]
        total_weight = sum(weights)
        if total_weight <= 0:
            continue
        k = min(cfg.evals_per_tick, len(candidates))
        picks = model.np_rng.choice(
            len(candidates), size=k, replace=False, p=[w / total_weight for w in weights]
        )
        wage = _mean_wage(firm, model)
        for idx in picks:
            task_id = candidates[int(idx)]
            quality = model.capability.task_quality(model.dataset.tasks[task_id])
            if quality <= 0.0 or wage <= 0.0:
                continue
            unit_ai_cost = ai_price / quality + cfg.adjustment_cost / 12.0
            roi = (wage - unit_ai_cost) / wage
            p = adoption_probability(model, firm, roi=roi, peer_share=peer_shares[firm.sector])
            if model.np_rng.uniform() < p:
                firm.adopted.add(task_id)
