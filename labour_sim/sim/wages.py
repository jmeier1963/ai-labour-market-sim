"""Wage formation: marginal products, bargained hire wages, on-the-job drift."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from labour_sim.sim.model import LabourMarketModel
    from labour_sim.sim.workers import WorkerAgent

WAGE_DRIFT_SPEED = 0.03  # employed wages close 3%/tick of the gap to marginal product


def effective_skill(model: "LabourMarketModel", worker: "WorkerAgent", occupation: str) -> float:
    """Skill carried into `occupation`, discounted by task distance; the
    retraining subsidy offsets part of the discount."""
    distance = model.occupation_distance(worker.occupation, occupation)
    if distance == 0.0:
        return worker.skill
    base = model.cfg.labour.retraining_skill_discount
    subsidy = model.cfg.policy.retraining_subsidy
    keep = base + (1.0 - base) * subsidy
    return worker.skill * keep**distance


def marginal_product(
    model: "LabourMarketModel", worker: "WorkerAgent", occupation: str, firm=None
) -> float:
    """Base productivity, scaled down by the share of the occupation's tasks the
    firm has automated and up by AI augmentation of the remaining tasks."""
    mp = model.dataset.occupations[occupation].base_wage * effective_skill(
        model, worker, occupation
    )
    if firm is not None:
        auto, aug = firm.occupation_ai_effects(model, occupation)
        mp *= max(0.05, 1.0 - auto) * (1.0 + aug)
    return mp


def hire_wage(model: "LabourMarketModel", worker: "WorkerAgent", occupation: str, firm) -> float:
    """Nash-style split between marginal product and the worker's reservation."""
    beta = model.cfg.matching.bargaining_beta
    mp = marginal_product(model, worker, occupation, firm)
    wage = beta * mp + (1.0 - beta) * worker.reservation_wage
    floor = model.cfg.policy.benefit_level * model.median_wage0
    return max(wage, worker.reservation_wage, floor)


def update_employed_wages(model: "LabourMarketModel") -> None:
    for worker in model.workers:
        if worker.state != "employed":
            continue
        mp = marginal_product(model, worker, worker.occupation, worker.employer)
        worker.wage += WAGE_DRIFT_SPEED * (mp - worker.wage)
        worker.reservation_wage = 0.9 * worker.wage
        worker.tenure += 1
