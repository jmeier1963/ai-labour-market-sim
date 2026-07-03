"""AI capability frontier: logistic growth with seeded AR(1) shocks, and the
mapping from frontier level to per-task automation quality."""

import numpy as np

from labour_sim.config import CapabilityConfig
from labour_sim.dataset import TaskDef

E1_PENALTY = 0.75  # E1 tasks need extra scaffolding; effective capability is reduced


class CapabilityProcess:
    def __init__(self, cfg: CapabilityConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self.level = cfg.initial
        self._shock = 0.0

    def step(self) -> None:
        cfg = self.cfg
        if cfg.ceiling > 0:
            drift = cfg.growth_rate * self.level * (1.0 - self.level / cfg.ceiling)
        else:
            drift = 0.0
        self._shock = cfg.noise_rho * self._shock + self.rng.normal(0.0, cfg.noise_sigma)
        self.level = float(np.clip(self.level + drift + self._shock * drift, 0.0, cfg.ceiling))

    @property
    def aug_level(self) -> float:
        """Capability gained since t0. The starting economy is calibrated as an
        equilibrium that already embeds initial capability, so augmentation
        effects respond only to growth since then."""
        return max(0.0, self.level - self.cfg.initial)

    def task_quality(self, task: TaskDef) -> float:
        """Quality of AI on this task in [0, 1]; 0 until capability clears difficulty."""
        if task.exposure == 0:
            return 0.0
        effective = self.level * (1.0 if task.exposure == 2 else E1_PENALTY)
        if task.difficulty >= 1.0:
            return 0.0
        return float(np.clip((effective - task.difficulty) / (1.0 - task.difficulty), 0.0, 1.0))
