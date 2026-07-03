"""Monte Carlo batches and parameter sweeps over the labour-market model.

Sub-run seeds derive deterministically from the master seed, so a batch is
reproducible as a whole and any single run can be reproduced in isolation.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from labour_sim.config import SimConfig

PERCENTILES = (5, 25, 50, 75, 95)

SUMMARY_STATS = {
    "peak_unemployment": lambda f: float(f["unemployment_rate"].max()),
    "peak_distress": lambda f: float((f["unemployment_rate"] + f["discouraged_share"]).max()),
    "final_unemployment": lambda f: float(f["unemployment_rate"].iloc[-1]),
    "final_output": lambda f: float(f["output_index"].iloc[-1]),
    "final_adoption": lambda f: float(f["adoption_share"].iloc[-1]),
    "final_gini": lambda f: float(f["wage_gini"].iloc[-1]),
}


def run_single(config_json: str) -> pd.DataFrame:
    """Top-level (picklable) worker: one full run -> model-vars DataFrame."""
    from labour_sim.sim.model import LabourMarketModel

    model = LabourMarketModel(SimConfig.model_validate_json(config_json))
    model.run()
    return model.datacollector.get_model_vars_dataframe()


def sub_seed(master_seed: int, run_index: int) -> int:
    return int(np.random.SeedSequence([master_seed, run_index]).generate_state(1)[0] % 2**31)


@dataclass
class BatchResult:
    config: SimConfig
    runs: list[pd.DataFrame] = field(default_factory=list)

    def bands(self, metric: str) -> pd.DataFrame:
        matrix = np.column_stack([frame[metric].to_numpy() for frame in self.runs])
        data = {
            f"p{p}": np.percentile(matrix, p, axis=1) for p in PERCENTILES
        }
        return pd.DataFrame(data, index=self.runs[0].index)

    def percentile_csv(self, metrics: list[str] | None = None) -> str:
        metrics = metrics or list(self.runs[0].columns)
        parts = []
        for metric in metrics:
            bands = self.bands(metric)
            bands.columns = [f"{metric}_{c}" for c in bands.columns]
            parts.append(bands)
        return pd.concat(parts, axis=1).to_csv(index_label="tick")


def monte_carlo(
    config: SimConfig,
    runs: int,
    processes: int | None = None,
    on_progress=None,
) -> BatchResult:
    """N runs with derived sub-seeds.

    Default (processes=None or 1) runs sequentially — safe from any context,
    including the Solara server thread. Pass processes>1 ONLY from an
    import-safe entry point (plain script, pytest): worker processes re-import
    __main__, and non-import-safe hosts (e.g. `solara run`) deadlock the pool.
    """
    configs = [
        config.model_copy(update={"seed": sub_seed(config.seed, k)}).model_dump_json()
        for k in range(runs)
    ]
    if processes is None or processes == 1:
        frames = []
        for k, payload in enumerate(configs):
            frames.append(run_single(payload))
            if on_progress is not None:
                on_progress(k + 1, runs)
    else:
        with ProcessPoolExecutor(max_workers=processes) as pool:
            frames = list(pool.map(run_single, configs))
        if on_progress is not None:
            on_progress(runs, runs)
    return BatchResult(config=config, runs=frames)


def set_by_path(config: SimConfig, path: str, value: float) -> SimConfig:
    """Return a new SimConfig with the dot-path leaf replaced (validated)."""
    data = config.model_dump()
    node = data
    keys = path.split(".")
    for key in keys[:-1]:
        if key not in node:
            raise KeyError(f"unknown config path {path!r}")
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f"unknown config path {path!r}")
    node[keys[-1]] = value
    return SimConfig.model_validate(data)


def summary_stat(frame: pd.DataFrame, name: str) -> float:
    if name not in SUMMARY_STATS:
        raise KeyError(f"unknown stat {name!r}; available: {sorted(SUMMARY_STATS)}")
    return SUMMARY_STATS[name](frame)


def sweep_1d(
    config: SimConfig,
    path: str,
    values: list[float],
    stat: str = "peak_unemployment",
    runs_per_point: int = 3,
    processes: int | None = None,
) -> pd.DataFrame:
    rows = []
    for value in values:
        result = monte_carlo(set_by_path(config, path, value), runs_per_point, processes)
        stats = [summary_stat(frame, stat) for frame in result.runs]
        rows.append(
            {
                "value": value,
                "stat_median": float(np.median(stats)),
                "stat_p25": float(np.percentile(stats, 25)),
                "stat_p75": float(np.percentile(stats, 75)),
            }
        )
    return pd.DataFrame(rows)


def sweep_2d(
    config: SimConfig,
    path_x: str,
    values_x: list[float],
    path_y: str,
    values_y: list[float],
    stat: str = "peak_unemployment",
    runs_per_point: int = 2,
    processes: int | None = None,
) -> pd.DataFrame:
    """Grid of median summary stats; rows indexed by y values, columns by x."""
    grid = np.zeros((len(values_y), len(values_x)))
    for i, vy in enumerate(values_y):
        cfg_y = set_by_path(config, path_y, vy)
        for j, vx in enumerate(values_x):
            result = monte_carlo(set_by_path(cfg_y, path_x, vx), runs_per_point, processes)
            grid[i, j] = np.median([summary_stat(f, stat) for f in result.runs])
    return pd.DataFrame(grid, index=values_y, columns=values_x)
