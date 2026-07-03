"""Firm demography: sustained low demand forces exit; profitable sectors attract
AI-native entrants — lower adoption hurdles AND a one-time evaluation of the
full current technology menu at founding — a vintage effect that accelerates
tipping and lets new firms be born with post-AI organizational structures."""

from typing import TYPE_CHECKING

from labour_sim.sim.adoption import _mean_wage, adoption_probability, sector_adoption_shares
from labour_sim.sim.firms import FirmAgent

if TYPE_CHECKING:
    from labour_sim.sim.model import LabourMarketModel

EXIT_DEMAND_THRESHOLD = 0.7  # demand persistently below 70% of entry-normalized level
ENTRY_SECTOR_BOOM = 1.1  # sector demand index above this accelerates entry


def update_entry_exit(model: "LabourMarketModel") -> None:
    """Exit on sustained demand shortfall; entry replaces exits (keeping the
    stationary baseline stationary) plus opportunity entry when sector demand
    booms. Entrants are AI-native (vintage effect)."""
    cfg = model.cfg.entry_exit

    exits_by_sector: dict[str, int] = {}
    for firm in list(model.firms):
        if firm.demand_factor < EXIT_DEMAND_THRESHOLD:
            firm.loss_streak += 1
        else:
            firm.loss_streak = 0
        firm.age += 1
        if firm.loss_streak >= cfg.exit_loss_ticks and len(model.firms) > len(
            model.dataset.sectors
        ):
            for worker in list(firm.employees):
                worker.lose_job()
            exits_by_sector[firm.sector] = exits_by_sector.get(firm.sector, 0) + 1
            firm.remove()

    by_sector: dict[str, int] = {}
    for firm in model.firms:
        by_sector[firm.sector] = by_sector.get(firm.sector, 0) + 1
    for sector_id, count in by_sector.items():
        entrants = exits_by_sector.get(sector_id, 0)
        if model.sector_demand.get(sector_id, 1.0) > ENTRY_SECTOR_BOOM:
            entrants += model.np_rng.poisson(cfg.entry_rate * count)
        for _ in range(entrants):
            _spawn_entrant(model, sector_id)


def _spawn_entrant(model: "LabourMarketModel", sector_id: str) -> None:
    cfg = model.cfg.adoption
    sector = model.dataset.sectors[sector_id]
    hurdle = max(
        0.0,
        model.np_rng.normal(
            cfg.hurdle_mean * (1.0 - cfg.entrant_hurdle_discount), cfg.hurdle_sigma
        ),
    )
    firm = FirmAgent(model, sector=sector_id, adoption_hurdle=float(hurdle))

    # Sample the entrant's composition from the sector mix so small firms are
    # statistically representative (a greedy largest-weight fill would never
    # allocate low-share occupations such as senior variants).
    size = sector.firm_size_min
    occupations = list(sector.occupation_mix)
    weights = [sector.occupation_mix[o] for o in occupations]
    draws = model.np_rng.choice(len(occupations), size=size, p=weights)
    targets: dict[str, int] = {}
    for index in draws:
        occupation = occupations[int(index)]
        targets[occupation] = targets.get(occupation, 0) + 1
    firm.base_targets = targets
    firm.target_occupations = dict(targets)

    # AI-native founding: unlike incumbents, who discover the technology menu
    # a few sampled tasks per month, a new firm evaluates the entire current
    # menu once at birth — it never builds the organizational structure that
    # yesterday's technology implied.
    adoption_cfg = model.cfg.adoption
    peer_share = sector_adoption_shares(model).get(sector_id, 0.0)
    wage = _mean_wage(firm, model)
    for task_id in firm.task_profile(model.dataset):
        quality = model.capability.task_quality(model.dataset.tasks[task_id])
        if quality <= 0.0 or wage <= 0.0:
            continue
        unit_ai_cost = model.ai_price / quality + adoption_cfg.adjustment_cost / 12.0
        roi = (wage - unit_ai_cost) / wage
        p = adoption_probability(model, firm, roi=roi, peer_share=peer_share)
        if model.np_rng.uniform() < p:
            firm.adopted.add(task_id)
