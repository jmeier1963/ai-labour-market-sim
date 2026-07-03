"""Research page: Monte Carlo bands, scenario save/load, run comparison, sweeps.

State is module-level (single-researcher tool); heavy work runs in a thread so
the UI stays responsive.
"""

import datetime

import matplotlib.pyplot as plt
import solara
import solara.lab

from labour_sim.batch import SUMMARY_STATS, BatchResult, monte_carlo, sweep_1d
from labour_sim.config import list_presets, load_preset
from labour_sim.scenarios import Scenario, scenario_from_json, scenario_to_json

BAND_METRICS = [
    "unemployment_rate",
    "discouraged_share",
    "adoption_share",
    "automated_task_share",
    "output_index",
    "wage_gini",
]
SWEEP_PATHS = [
    "capability.growth_rate",
    "ai_cost.decline_rate",
    "adoption.imitation_weight",
    "matching.efficiency",
    "policy.benefit_level",
    "policy.firing_cost",
]

preset = solara.reactive("baseline")
n_runs = solara.reactive(20)
n_workers = solara.reactive(1500)
status = solara.reactive("idle")
results: solara.Reactive[dict[str, BatchResult]] = solara.reactive({})
selected_a = solara.reactive("")
selected_b = solara.reactive("")
metric = solara.reactive("unemployment_rate")
loaded_scenario: solara.Reactive[Scenario | None] = solara.reactive(None)

sweep_path = solara.reactive(SWEEP_PATHS[0])
sweep_stat = solara.reactive("peak_unemployment")
sweep_lo = solara.reactive(0.0)
sweep_hi = solara.reactive(0.1)
sweep_steps = solara.reactive(5)
sweep_table = solara.reactive(None)
sweep_status = solara.reactive("idle")


def _current_config():
    scenario = loaded_scenario.value
    base = scenario.config if scenario else load_preset(preset.value)
    return base.model_copy(update={"n_workers": n_workers.value})


# solara.lab.task runs the work in a thread that keeps the session's kernel
# context bound — reactive updates from a plain threading.Thread would land in
# the default context and never reach the browser session.
@solara.lab.task
def _run_batch() -> None:
    label = loaded_scenario.value.name if loaded_scenario.value else preset.value
    key = f"{label} ({n_runs.value} runs, N={n_workers.value})"
    try:
        status.value = f"running {key} ..."

        def progress(done: int, total: int) -> None:
            status.value = f"running {key}: {done}/{total}"

        result = monte_carlo(_current_config(), runs=n_runs.value, on_progress=progress)
        results.value = {**results.value, key: result}
        if not selected_a.value:
            selected_a.value = key
        elif not selected_b.value:
            selected_b.value = key
        status.value = f"done: {key}"
    except Exception as error:  # surface, never swallow
        status.value = f"error: {error}"


@solara.lab.task
def _run_sweep() -> None:
    try:
        sweep_status.value = "sweeping ..."
        steps = max(2, int(sweep_steps.value))
        span = (sweep_hi.value - sweep_lo.value) or 1.0
        values = [round(sweep_lo.value + span * i / (steps - 1), 6) for i in range(steps)]
        sweep_table.value = sweep_1d(
            _current_config(), sweep_path.value, values, stat=sweep_stat.value
        )
        sweep_status.value = "sweep done"
    except Exception as error:
        sweep_status.value = f"error: {error}"


def _band_figure():
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    palette = {"A": "tab:blue", "B": "tab:red"}
    for slot, key in (("A", selected_a.value), ("B", selected_b.value)):
        result = results.value.get(key)
        if result is None:
            continue
        bands = result.bands(metric.value)
        color = palette[slot]
        ax.fill_between(bands.index, bands["p5"], bands["p95"], alpha=0.15, color=color)
        ax.fill_between(bands.index, bands["p25"], bands["p75"], alpha=0.3, color=color)
        ax.plot(bands.index, bands["p50"], color=color, label=f"{slot}: {key}")
    ax.set_xlabel("month")
    ax.set_ylabel(metric.value)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


@solara.component
def ResearchPage() -> None:
    solara.Title("Research — Monte Carlo & sweeps")
    with solara.Columns([1, 2]):
        with solara.Column():
            with solara.Card("Batch setup"):
                solara.Select("Preset", value=preset, values=list_presets())
                solara.SliderInt("Runs", value=n_runs, min=4, max=100)
                solara.SliderInt("Workers per run", value=n_workers, min=500, max=5000, step=500)
                solara.Button("Run Monte Carlo", on_click=_run_batch, color="primary")
                solara.Text(status.value)
            with solara.Card("Scenario file"):
                if loaded_scenario.value:
                    solara.Text(f"loaded: {loaded_scenario.value.name}")
                    solara.Button("Clear", on_click=lambda: loaded_scenario.set(None))

                def on_file(file_info) -> None:
                    try:
                        loaded_scenario.value = scenario_from_json(
                            file_info["file_obj"].read().decode("utf-8")
                        )
                        status.value = f"scenario loaded: {loaded_scenario.value.name}"
                    except ValueError as error:
                        status.value = str(error)

                solara.FileDrop(label="Drop scenario JSON here", on_file=on_file, lazy=False)

                def scenario_json() -> str:
                    scenario = Scenario(
                        name=preset.value,
                        config=_current_config(),
                        created=datetime.date.today().isoformat(),
                    )
                    return scenario_to_json(scenario)

                solara.FileDownload(
                    scenario_json, filename="scenario.json", label="Save current as scenario"
                )
        with solara.Column():
            with solara.Card("Uncertainty bands & comparison"):
                keys = list(results.value)
                solara.Select("Metric", value=metric, values=BAND_METRICS)
                solara.Select("Run A", value=selected_a, values=[""] + keys)
                solara.Select("Run B (overlay)", value=selected_b, values=[""] + keys)
                if selected_a.value or selected_b.value:
                    fig = _band_figure()
                    solara.FigureMatplotlib(fig)
                    plt.close(fig)
                    result = results.value.get(selected_a.value)
                    if result is not None:
                        solara.FileDownload(
                            lambda: result.percentile_csv(),
                            filename="batch_percentiles.csv",
                            label="Download percentile CSV",
                        )
            with solara.Card("1-D parameter sweep"):
                solara.Select("Parameter", value=sweep_path, values=SWEEP_PATHS)
                solara.Select("Statistic", value=sweep_stat, values=sorted(SUMMARY_STATS))
                solara.InputFloat("From", value=sweep_lo)
                solara.InputFloat("To", value=sweep_hi)
                solara.SliderInt("Grid points", value=sweep_steps, min=2, max=12)
                solara.Button("Run sweep", on_click=_run_sweep, color="primary")
                solara.Text(sweep_status.value)
                if sweep_table.value is not None:
                    table = sweep_table.value
                    fig, ax = plt.subplots(figsize=(8.5, 3.6))
                    ax.fill_between(
                        table["value"], table["stat_p25"], table["stat_p75"], alpha=0.25
                    )
                    ax.plot(table["value"], table["stat_median"], marker="o")
                    ax.set_xlabel(sweep_path.value)
                    ax.set_ylabel(sweep_stat.value)
                    ax.grid(alpha=0.25)
                    fig.tight_layout()
                    solara.FigureMatplotlib(fig)
                    plt.close(fig)
