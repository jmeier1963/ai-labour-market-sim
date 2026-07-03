"""Model invariants checked by tests after every step (conservation contracts)."""

import math
from typing import TYPE_CHECKING

from labour_sim.sim.workers import EMPLOYED, WORKER_STATES

if TYPE_CHECKING:
    from labour_sim.sim.model import LabourMarketModel


def check_invariants(model: "LabourMarketModel") -> None:
    workers = list(model.workers)
    firms = list(model.firms)
    firm_set = set(firms)

    assert len(workers) == model.cfg.n_workers, "worker count must be conserved"

    for worker in workers:
        assert worker.state in WORKER_STATES, f"unknown state {worker.state!r}"
        if worker.state == EMPLOYED:
            assert worker.employer is not None, "employed worker without employer"
            assert worker.employer in firm_set, "employed at a firm not in the model"
            assert worker in worker.employer.employees, "employer/employee link broken"
            assert worker.wage > 0, "employed worker must earn a positive wage"
        else:
            assert worker.employer is None, "non-employed worker keeps an employer"
        assert math.isfinite(worker.wage) and worker.wage >= 0
        assert math.isfinite(worker.reservation_wage) and worker.reservation_wage >= 0

    seen: set[int] = set()
    for firm in firms:
        for worker in firm.employees:
            assert worker.employer is firm, "employee points at a different firm"
            assert id(worker) not in seen, "worker employed by two firms"
            seen.add(id(worker))
