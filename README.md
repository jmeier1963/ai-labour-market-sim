# AI Labour Market Simulator (Agent-Based Model)

An agent-based simulator of AI adoption dynamics at the **organizational** and
**labour-market** level. Individual firms decide task-by-task AI adoption on
ROI and peer imitation; individual workers search, match, retrain, and bargain
over wages. Tipping points, S-curves, displacement waves, wage scarring, and
inequality dynamics **emerge** from micro-interactions — nothing is a sigmoid
of time.

Built with [Mesa 3](https://mesa.readthedocs.io/) (ABM engine) and
[Solara](https://solara.dev/) (interactive dashboard). The previous aggregate
system-dynamics app is preserved in `legacy/`.

## Requirements and installation

- Python ≥ 3.12 (pinned to 3.13 via `.python-version`)
- [uv](https://docs.astral.sh/uv/) — manages the virtualenv and dependencies

```sh
uv sync          # creates .venv and installs everything from pyproject.toml
```

## Running the app

```sh
uv run solara run app.py
# open http://localhost:8765
```

The first page load takes ~10 s (the model synthesizes its population). The
navigation bar has two pages: **Home** (the live model) and **Research**
(batch experiments).

Useful flags: `--port 8080` to change the port; `--production` to disable
hot-reload (recommended when running long batches, since a code reload
mid-batch detaches running jobs from the UI).

## The Live model page

The left sidebar holds the controls, the main area the visualizations.

**Controls.** `RESET` rebuilds the model from the current parameters; `▶`
plays continuously (Play Interval sets the delay per tick); `STEP` advances
one month. Changing any model parameter triggers a rebuild, so set parameters
first, then play. One tick = one month; a typical horizon is 240 ticks
(20 years).

**Model parameters.**

| Control | Meaning | Config path |
|---|---|---|
| Preset scenario | Base configuration; **pins its own parameters** (sliders control the rest). Use `baseline` for full slider control. | — |
| Random seed | Master seed; identical seeds reproduce identical runs bit-for-bit | `seed` |
| Workers | Population size (firms scale with it, roughly 1 firm per 10 workers) | `n_workers` |
| AI capability growth /mo | Logistic growth rate of the capability frontier | `capability.growth_rate` |
| AI cost decline /mo | Exponential decline rate of the AI price | `ai_cost.decline_rate` |
| Adoption imitation | Weight of sector peer adoption in firms' decisions | `adoption.imitation_weight` |
| Benefit level | Reservation-wage floor as a fraction of the t0 median wage | `policy.benefit_level` |
| Retraining subsidy | Offsets the skill discount when switching occupations | `policy.retraining_subsidy` |

**Reading the agent-space view.** Each sector is a cluster on a circle. Firms
are squares — size scales with headcount, colour shows the share of the firm's
task mass performed by AI (dark purple = none, yellow = high, `plasma`
colormap). Workers are dots: teal = employed (drawn around their employer),
orange = searching (they drift to the ring at the centre), grey = discouraged
(the innermost cloud). During a fast takeoff you can watch the adoption
cascade as a colour front sweeping through sector clusters — manufacturing
stays dark because its manual (E0) tasks resist automation.

**Chart panels.** Unemployment + discouraged share; adoption share, automated
task share, and capability; output index + mean wage; wage Gini; a
wage-percentile fan (p10–p90); the Beveridge-curve trace (colour = time); and
the cumulative occupation-switch matrix (SOC group → SOC group at hire).

**Export.** The Export card downloads the full tick-level metrics of the
current run as CSV (one row per month, one column per metric — see
`labour_sim/sim/metrics.py` for definitions).

## The Research page

For claims you intend to rely on, use this page rather than single live runs —
tipping dynamics are path-dependent and seed-to-seed variance is large.

**Monte Carlo batches.** Choose a preset, number of runs, and workers per run,
then `RUN MONTE CARLO`. Runs execute sequentially with a progress readout;
expect roughly 5–8 s per run at N=1500 over 20 years (20 runs ≈ 2–3 minutes).
Each finished batch is added to the run registry.

**Uncertainty bands & comparison.** Pick a metric and a registry entry as
Run A; the chart shows the p5–p95 and p25–p75 bands with the median. Select a
second entry as Run B to overlay two experiments (e.g. `fast_takeoff` vs
`policy_cushion`). `DOWNLOAD PERCENTILE CSV` exports all metrics' percentile
bands for the selected batch.

**Scenario files.** `SAVE CURRENT AS SCENARIO` downloads a versioned JSON
snapshot of the full configuration (validated by Pydantic on load). Drop a
scenario JSON onto the drop zone to load it; subsequent batches then use the
scenario's configuration (with the Workers slider still applied). Scenario
files are the way to archive an experiment alongside exported results so any
figure can be regenerated later.

**1-D parameter sweeps.** Choose a dot-path parameter (e.g.
`capability.growth_rate`), a summary statistic (e.g. `peak_unemployment`,
`peak_distress`, `final_gini`), a value range, and grid points. The sweep runs
3 seeds per grid point and plots the median with an interquartile band.
Sweeps cost `points × 3` full runs — size accordingly.

## Presets

| Preset | Story |
|---|---|
| `baseline` | Moderate capability growth and frictions; adoption cascade around year 4 |
| `fast_takeoff` | 8%/mo capability growth, fast cost decline, strong imitation; cascade around year 2 |
| `high_friction` | Brisk AI progress into a rigid labour market: slow matching, weak safety net, slow retraining |
| `policy_cushion` | The fast-takeoff AI path plus a strong safety net and heavily subsidised retraining |

Presets live in `labour_sim/data/presets/*.json` as overrides on the default
`SimConfig`; add a new file there to define your own (it appears in both
pages' dropdowns automatically).

## Using the engine from Python

The engine is UI-independent — drive it from scripts or notebooks:

```python
from labour_sim.config import SimConfig, load_preset
from labour_sim.sim.model import LabourMarketModel

# Single run
model = LabourMarketModel(SimConfig(seed=42, n_workers=2000, horizon_years=20))
model.run()
df = model.datacollector.get_model_vars_dataframe()   # tick-level metrics
df.to_csv("run.csv")

# Preset + targeted override (validated)
from labour_sim.batch import set_by_path
cfg = set_by_path(load_preset("fast_takeoff"), "policy.benefit_level", 0.65)

# Monte Carlo with percentile bands
from labour_sim.batch import monte_carlo
result = monte_carlo(cfg, runs=24, processes=6)       # parallel OK in scripts
bands = result.bands("unemployment_rate")             # p5/p25/p50/p75/p95
print(result.percentile_csv())

# Parameter sweeps
from labour_sim.batch import sweep_1d, sweep_2d
table = sweep_1d(cfg, "matching.efficiency", [0.25, 0.45, 0.65],
                 stat="peak_distress", runs_per_point=5)
grid = sweep_2d(cfg, "capability.growth_rate", [0.02, 0.05, 0.08],
                "matching.efficiency", [0.25, 0.45, 0.65],
                stat="peak_distress")

# Scenario files
from labour_sim.scenarios import Scenario, scenario_to_json, scenario_from_json
payload = scenario_to_json(Scenario(name="my-experiment", config=cfg))
```

All `SimConfig` fields and their meanings are documented inline in
[labour_sim/config.py](labour_sim/config.py); the tick-by-tick model logic is
described in [docs/model.md](docs/model.md).

⚠️ `monte_carlo(..., processes>1)` uses process pools and must be called from
an import-safe entry point (a plain script guarded by
`if __name__ == "__main__":`, or pytest). Inside `solara run` it would
deadlock — the UI therefore always runs batches sequentially.

## Reproducibility

One master seed drives everything: independent numpy streams per subsystem
(synthesis, matching, capability shocks) and derived sub-seeds per Monte Carlo
run. Identical seed → identical series, bit for bit — this contract is
enforced by tests. Exported scenario JSON + seed is sufficient to regenerate
any result exactly.

## Testing and development

```sh
uv run pytest              # full suite (~3 min): unit, invariant, calibration gates
uv run pytest --cov        # with coverage (≥80% enforced on the engine)
uv run pytest tests/test_matching.py   # single module while developing
uv run ruff check labour_sim tests app.py
```

The suite includes **calibration gates**: the no-AI baseline must hold
empirical labour-flow anchors (unemployment 3–8%, monthly job-finding 10–45%,
separations 0.6–2.5%, Gini 0.20–0.45, no employment drift). Any new mechanism
must keep these green — treat a red gate as a modelling error, not a test to
adjust. Policy-lever signs are deliberately **not** gated (they are seed- and
specification-sensitive; see the paper); only mechanically-signed relations
(matching efficiency) are enforced.

## Paper

A ten-page write-up of the model design and emergent outcomes:
[paper/paper.pdf](paper/paper.pdf).

```sh
uv run python paper/make_figures.py                    # regenerate all result figures
uv run solara run app.py --production &                # then, for the screenshots:
uv run --with playwright --with pillow python paper/take_screenshots.py
cd paper && xelatex paper.tex && bibtex paper && xelatex paper.tex && xelatex paper.tex
```

## Data

Task exposure classes follow Eloundou et al. (2023) E0/E1/E2; occupations are
SOC major groups; sectors carry over from the legacy model. **All bundled
values are placeholders** pending curation — every record carries a `source`
field, and [docs/data-sources.md](docs/data-sources.md) maps each field to its
curation target. To experiment with your own assumptions, edit the JSON files
in `labour_sim/data/` (weights and shares are normalized at load, so they only
need to be approximately consistent).

## Troubleshooting

- **Port already in use** — a previous server is still running:
  `pkill -f "solara run app.py"`, or pass `--port`.
- **Batch finishes but the UI never updates** — you are running with
  hot-reload and edited code mid-batch; restart with `--production`.
- **Slow first tick after Reset** — population synthesis at large N; reduce
  the Workers slider for interactive exploration (results are scale-robust in
  our checks down to ~1000).
- **Everything is deterministic but I want variation** — change the seed;
  same-seed runs are identical by design.

## Layout

```
app.py                  Solara entry (Live + Research pages)
labour_sim/config.py    Pydantic SimConfig + presets (data/presets/*.json)
labour_sim/dataset.py   task/occupation/sector schemas + loader
labour_sim/sim/         engine: model, agents, capability, adoption, demand,
                        labour demand, matching, wages, flows, entry/exit
labour_sim/batch.py     Monte Carlo, percentile bands, parameter sweeps
labour_sim/scenarios.py versioned scenario files
labour_sim/viz/         dashboard, agent space, diagnostics, research page
tests/                  unit + invariant + calibration-gate tests
docs/                   model.md (tick order, findings), data-sources.md
paper/                  ten-page paper + figure/screenshot generation scripts
legacy/                 the original vanilla-JS aggregate simulator
```
