"""Bundled empirical data: tasks, occupations, sectors.

Values ship as clearly-labelled placeholders in the correct schema; every record
carries a `source` field. See docs/data-sources.md for provenance and curation
status. Weights and shares are normalized at load so the JSON only needs to be
approximately consistent.
"""

import json
import math
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_DATA_DIR = Path(__file__).parent / "data"


class TaskDef(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    exposure: int = Field(ge=0, le=2)  # Eloundou et al. E0 / E1 / E2
    difficulty: float = Field(ge=0, le=1)  # capability needed before automatable
    augmentation: float = Field(ge=0, le=1)  # human productivity boost when AI assists
    source: str


class OccupationDef(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    base_wage: float = Field(gt=0)  # relative units, median occupation ~ 1
    employment_share: float = Field(gt=0, le=1)
    task_weights: dict[str, float]
    source: str


class SectorDef(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    employment_share: float = Field(gt=0, le=1)
    demand_elasticity: float = Field(ge=0, le=2)
    occupation_mix: dict[str, float]
    firm_size_alpha: float = Field(gt=1)  # Pareto tail of firm sizes
    firm_size_min: int = Field(ge=1)
    source: str


class Dataset(BaseModel):
    model_config = ConfigDict(frozen=True)

    tasks: dict[str, TaskDef]
    occupations: dict[str, OccupationDef]
    sectors: dict[str, SectorDef]

    def occupation_distance(self, a: str, b: str) -> float:
        """1 - cosine similarity of task-weight vectors; retraining feasibility."""
        wa, wb = self.occupations[a].task_weights, self.occupations[b].task_weights
        dot = sum(wa[t] * wb.get(t, 0.0) for t in wa)
        norm = math.sqrt(sum(v * v for v in wa.values())) * math.sqrt(
            sum(v * v for v in wb.values())
        )
        return 1.0 - dot / norm if norm else 1.0


def _normalized(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return {k: v / total for k, v in weights.items()}


def _load_json(name: str) -> dict:
    return json.loads((_DATA_DIR / name).read_text(encoding="utf-8"))


@lru_cache
def load_dataset() -> Dataset:
    tasks = {
        record["id"]: TaskDef.model_validate(record) for record in _load_json("tasks.json")["tasks"]
    }

    occ_records = _load_json("occupations.json")["occupations"]
    share_total = sum(r["employment_share"] for r in occ_records)
    occupations = {}
    for record in occ_records:
        unknown = set(record["task_weights"]) - set(tasks)
        if unknown:
            raise ValueError(f"occupation {record['id']} references unknown tasks {unknown}")
        occupations[record["id"]] = OccupationDef.model_validate(
            {
                **record,
                "employment_share": record["employment_share"] / share_total,
                "task_weights": _normalized(record["task_weights"]),
            }
        )

    sector_records = _load_json("sectors.json")["sectors"]
    sector_total = sum(r["employment_share"] for r in sector_records)
    sectors = {}
    for record in sector_records:
        unknown = set(record["occupation_mix"]) - set(occupations)
        if unknown:
            raise ValueError(f"sector {record['id']} references unknown occupations {unknown}")
        sectors[record["id"]] = SectorDef.model_validate(
            {
                **record,
                "employment_share": record["employment_share"] / sector_total,
                "occupation_mix": _normalized(record["occupation_mix"]),
            }
        )

    return Dataset(tasks=tasks, occupations=occupations, sectors=sectors)
