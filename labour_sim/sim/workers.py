"""Worker agents: employment state, wages, and search attributes."""

from typing import TYPE_CHECKING

import mesa

if TYPE_CHECKING:
    from labour_sim.sim.firms import FirmAgent
    from labour_sim.sim.model import LabourMarketModel

EMPLOYED = "employed"
SEARCHING = "searching"
DISCOURAGED = "discouraged"

WORKER_STATES = (EMPLOYED, SEARCHING, DISCOURAGED)


class WorkerAgent(mesa.Agent):
    def __init__(
        self,
        model: "LabourMarketModel",
        occupation: str,
        skill: float,
    ) -> None:
        super().__init__(model)
        self.occupation = occupation
        self.skill = skill
        self.state: str = SEARCHING
        self.employer: "FirmAgent | None" = None
        self.wage: float = 0.0
        self.reservation_wage: float = 0.0
        self.unemployment_ticks: int = 0
        self.tenure: int = 0

    def start_job(self, firm: "FirmAgent", wage: float) -> None:
        self.state = EMPLOYED
        self.employer = firm
        self.wage = wage
        self.reservation_wage = 0.9 * wage
        self.unemployment_ticks = 0
        self.tenure = 0
        firm.employees.append(self)

    def lose_job(self) -> None:
        if self.employer is not None:
            self.employer.employees.remove(self)
        self.state = SEARCHING
        self.employer = None
        self.reservation_wage = max(self.reservation_wage, 0.9 * self.wage)
        self.wage = 0.0
        self.tenure = 0
