"""Firm demography: sustained low demand forces exit; profitable sectors attract
entrants that start AI-native (lower adoption hurdles), a vintage effect that
accelerates tipping."""

from typing import TYPE_CHECKING

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

    size = sector.firm_size_min
    mix = sorted(sector.occupation_mix.items(), key=lambda kv: kv[1], reverse=True)
    targets: dict[str, int] = {}
    remaining = size
    for occupation, weight in mix:
        n = min(remaining, max(1, round(size * weight)))
        if n <= 0:
            break
        targets[occupation] = n
        remaining -= n
        if remaining <= 0:
            break
    firm.base_targets = targets
    firm.target_occupations = dict(targets)
