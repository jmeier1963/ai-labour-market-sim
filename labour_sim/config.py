"""Simulation configuration (Pydantic) and named presets.

All rates are per monthly tick unless stated otherwise. Wages and costs are in
relative units where the economy-wide median starting wage is 1.0.
"""

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_PRESET_DIR = Path(__file__).parent / "data" / "presets"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CapabilityConfig(_Frozen):
    initial: float = Field(0.15, ge=0, le=1)
    ceiling: float = Field(0.95, ge=0, le=1)
    growth_rate: float = Field(0.035, ge=0, le=1)  # logistic rate toward ceiling
    noise_sigma: float = Field(0.008, ge=0, le=0.2)
    noise_rho: float = Field(0.6, ge=0, lt=1)  # AR(1) persistence of shocks


class AICostConfig(_Frozen):
    initial: float = Field(0.8, gt=0)  # cost per task-unit vs. median-wage labour cost
    decline_rate: float = Field(0.015, ge=0, le=0.5)  # exponential monthly decline


class AdoptionConfig(_Frozen):
    evals_per_tick: int = Field(4, ge=1)  # tasks a firm re-evaluates per tick
    logistic_slope: float = Field(6.0, gt=0)
    imitation_weight: float = Field(0.35, ge=0, le=2)
    adjustment_cost: float = Field(0.25, ge=0)  # one-off cost amortised into ROI
    hurdle_mean: float = Field(0.15, ge=0)
    hurdle_sigma: float = Field(0.10, ge=0)
    entrant_hurdle_discount: float = Field(0.5, ge=0, le=1)


class MatchingConfig(_Frozen):
    applications_per_searcher: int = Field(5, ge=1)
    efficiency: float = Field(0.45, gt=0, le=1)  # chance a staffed vacancy completes a hire per tick
    bargaining_beta: float = Field(0.5, ge=0, le=1)  # worker share of match surplus
    same_occupation_bias: float = Field(2.0, ge=0)
    reservation_decay: float = Field(0.01, ge=0, le=0.2)  # per tick unemployed
    discouragement_ticks: int = Field(24, ge=1)


class PolicyConfig(_Frozen):
    benefit_level: float = Field(0.45, ge=0, le=1)  # reservation floor vs. median wage
    retraining_subsidy: float = Field(0.3, ge=0, le=1)
    firing_cost: float = Field(0.2, ge=0)  # months of wage per layoff


class DemandConfig(_Frozen):
    growth_trend: float = Field(0.0, ge=0, le=0.05)  # secular monthly demand growth (0 = stationary baseline)
    price_sensitivity: float = Field(4.0, ge=0)  # logit slope for market shares


class EntryExitConfig(_Frozen):
    exit_loss_ticks: int = Field(9, ge=1)  # consecutive loss months before exit
    entry_rate: float = Field(0.004, ge=0, le=0.1)  # firms per incumbent per tick


class LabourConfig(_Frozen):
    separation_rate: float = Field(0.012, ge=0, le=0.2)  # baseline monthly quits
    initial_unemployment: float = Field(0.05, ge=0, le=0.5)
    retraining_wait_ticks: int = Field(6, ge=1)
    retraining_skill_discount: float = Field(0.6, ge=0, le=1)  # rho^distance base
    reentry_rate: float = Field(0.03, ge=0, le=1)  # discouraged -> searching per tick


class SimConfig(_Frozen):
    seed: int = Field(42, ge=0)
    n_workers: int = Field(5000, ge=500, le=50000)
    horizon_years: int = Field(20, ge=1, le=60)
    capability: CapabilityConfig = CapabilityConfig()
    ai_cost: AICostConfig = AICostConfig()
    adoption: AdoptionConfig = AdoptionConfig()
    matching: MatchingConfig = MatchingConfig()
    policy: PolicyConfig = PolicyConfig()
    demand: DemandConfig = DemandConfig()
    entry_exit: EntryExitConfig = EntryExitConfig()
    labour: LabourConfig = LabourConfig()

    @property
    def ticks(self) -> int:
        return self.horizon_years * 12


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@lru_cache
def _preset_files() -> dict[str, Path]:
    return {p.stem: p for p in sorted(_PRESET_DIR.glob("*.json"))}


def list_presets() -> list[str]:
    return list(_preset_files())


def load_preset(name: str) -> SimConfig:
    """Build a SimConfig from defaults plus the preset's overrides."""
    files = _preset_files()
    if name not in files:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(files)}")
    payload = json.loads(files[name].read_text(encoding="utf-8"))
    base = SimConfig().model_dump()
    return SimConfig.model_validate(_deep_merge(base, payload.get("overrides", {})))
