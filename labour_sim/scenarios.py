"""Scenario files: named, versioned, validated SimConfig snapshots with notes."""

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from labour_sim.config import SimConfig

SCHEMA_VERSION = 1


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    notes: str = ""
    config: SimConfig = SimConfig()
    schema_version: int = SCHEMA_VERSION
    created: str = ""  # ISO date; set by the UI at save time


def scenario_to_json(scenario: Scenario) -> str:
    return scenario.model_dump_json(indent=2)


def scenario_from_json(payload: str) -> Scenario:
    try:
        scenario = Scenario.model_validate_json(payload)
    except ValidationError as error:
        raise ValueError(f"invalid scenario file: {error}") from error
    if scenario.schema_version > SCHEMA_VERSION:
        raise ValueError(
            f"scenario schema v{scenario.schema_version} is newer than supported v{SCHEMA_VERSION}"
        )
    return scenario
