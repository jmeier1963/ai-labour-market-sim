"""LabourMarketModel: Mesa model orchestrating the per-tick processes.

Tick order (phase 3, no AI yet): separations -> vacancy posting -> matching ->
unemployed updates -> employed wage drift -> metric collection. AI capability,
adoption, demand, and entry/exit slot into this sequence in phase 4.
"""

import math

import mesa
import numpy as np

from labour_sim.config import SimConfig
from labour_sim.dataset import Dataset, load_dataset
from labour_sim.sim.adoption import update_adoption
from labour_sim.sim.capability import CapabilityProcess
from labour_sim.sim.demand import update_product_market
from labour_sim.sim.entry_exit import update_entry_exit
from labour_sim.sim.firms import FirmAgent
from labour_sim.sim.labour_demand import update_labour_demand
from labour_sim.sim.labour_flows import post_vacancies, separations, update_unemployed
from labour_sim.sim.matching import run_matching
from labour_sim.sim.metrics import model_reporters
from labour_sim.sim.synthesis import synthesize
from labour_sim.sim.wages import update_employed_wages
from labour_sim.sim.workers import EMPLOYED, SEARCHING, WorkerAgent


class LabourMarketModel(mesa.Model):
    def __init__(self, config: SimConfig | None = None, dataset: Dataset | None = None) -> None:
        self.cfg = config or SimConfig()
        super().__init__(rng=self.cfg.seed)
        self.dataset = dataset or load_dataset()
        # Independent, reproducible numpy streams per subsystem.
        seeds = np.random.SeedSequence(self.cfg.seed).spawn(3)
        self.np_rng = np.random.default_rng(seeds[0])
        self.match_rng = np.random.default_rng(seeds[1])
        self.capability = CapabilityProcess(self.cfg.capability, np.random.default_rng(seeds[2]))
        self.ai_price: float = self.cfg.ai_cost.initial
        self.sector_demand: dict[str, float] = {}

        self.median_wage0: float = 1.0  # set by synthesize()
        self._distance_cache: dict[tuple[str, str], float] = {}
        # Cumulative occupation switches at hire: (from, to) -> count.
        self.occupation_flows: dict[tuple[str, str], int] = {}
        self.flow_separations = 0
        self.flow_hires = 0
        self.flow_layoffs = 0
        self.flow_employed_start = 0
        self.flow_searchers = 0

        synthesize(self)
        # t0 task capacity per sector: anchors total sector volume in demand.py.
        self.sector_size_base: dict[str, float] = {}
        for firm in self.firms:
            size = sum(firm.base_targets.values())
            self.sector_size_base[firm.sector] = self.sector_size_base.get(firm.sector, 0.0) + size
        self.datacollector = mesa.DataCollector(model_reporters=model_reporters())

    @property
    def workers(self) -> mesa.agent.AgentSet:
        return self.agents_by_type[WorkerAgent]

    @property
    def firms(self) -> mesa.agent.AgentSet:
        return self.agents_by_type[FirmAgent]

    def occupation_distance(self, a: str, b: str) -> float:
        key = (a, b)
        if key not in self._distance_cache:
            self._distance_cache[key] = self.dataset.occupation_distance(a, b)
        return self._distance_cache[key]

    def step(self) -> None:
        self.flow_employed_start = sum(1 for w in self.workers if w.state == EMPLOYED)
        self.capability.step()
        self.ai_price = self.cfg.ai_cost.initial * math.exp(
            -self.cfg.ai_cost.decline_rate * self.steps
        )
        update_adoption(self)
        update_product_market(self)
        self.flow_layoffs = update_labour_demand(self)
        self.flow_separations = separations(self)
        post_vacancies(self)
        self.flow_searchers = sum(1 for w in self.workers if w.state == SEARCHING)
        self.flow_hires = run_matching(self)
        update_unemployed(self)
        update_employed_wages(self)
        update_entry_exit(self)
        self.datacollector.collect(self)

    def run(self, ticks: int | None = None) -> None:
        for _ in range(ticks if ticks is not None else self.cfg.ticks):
            self.step()
