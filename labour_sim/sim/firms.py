"""Firm agents: employment relations, vacancies, and AI adoption state."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mesa

if TYPE_CHECKING:
    from labour_sim.sim.model import LabourMarketModel
    from labour_sim.sim.workers import WorkerAgent


@dataclass
class Vacancy:
    occupation: str
    wage_offer: float
    firm: "FirmAgent"
    age: int = 0
    applicants: list = field(default_factory=list)


class FirmAgent(mesa.Agent):
    def __init__(
        self,
        model: "LabourMarketModel",
        sector: str,
        adoption_hurdle: float,
    ) -> None:
        super().__init__(model)
        self.sector = sector
        self.adoption_hurdle = adoption_hurdle
        self.employees: list["WorkerAgent"] = []
        self.vacancies: list[Vacancy] = []
        self.target_occupations: dict[str, int] = {}
        self.base_targets: dict[str, int] = {}
        self.adopted: set[str] = set()
        self.markup: float = 0.15
        self.price: float = 1.0
        self.demand_factor: float = 1.0
        self.output: float = 0.0
        self.loss_streak: int = 0
        self.age: int = 0
        self._ai_effects: dict[str, tuple[float, float]] = {}

    def reset_tick_cache(self) -> None:
        self._ai_effects.clear()

    def task_profile(self, dataset) -> dict[str, float]:
        """Task mass implied by the firm's target occupation composition."""
        profile: dict[str, float] = {}
        source = self.base_targets or self.target_occupations
        for occupation, count in source.items():
            for task_id, weight in dataset.occupations[occupation].task_weights.items():
                profile[task_id] = profile.get(task_id, 0.0) + weight * count
        return profile

    def occupation_ai_effects(self, model, occupation: str) -> tuple[float, float]:
        """(automated_share, augmentation_boost) for an occupation at this firm."""
        if occupation in self._ai_effects:
            return self._ai_effects[occupation]
        auto = 0.0
        aug = 0.0
        for task_id, weight in model.dataset.occupations[occupation].task_weights.items():
            task = model.dataset.tasks[task_id]
            quality = model.capability.task_quality(task)
            if task_id in self.adopted and quality > 0:
                auto += weight * quality
            elif task.exposure > 0:
                aug += weight * task.augmentation * model.capability.aug_level
        self._ai_effects[occupation] = (auto, aug)
        return auto, aug

    @property
    def headcount(self) -> int:
        return len(self.employees)

    def wage_bill(self) -> float:
        return sum(w.wage for w in self.employees)

    def adoption_share(self) -> float:
        """Share of the firm's task mass currently performed by AI."""
        if not self.employees:
            return 1.0 if self.adopted else 0.0
        dataset = self.model.dataset  # type: ignore[attr-defined]
        total = 0.0
        adopted = 0.0
        for worker in self.employees:
            weights = dataset.occupations[worker.occupation].task_weights
            for task_id, weight in weights.items():
                total += weight
                if task_id in self.adopted:
                    adopted += weight
        return adopted / total if total else 0.0
