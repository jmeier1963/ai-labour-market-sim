"""SolaraViz dashboard: interactive controls, live time-series, CSV export.

Run with: uv run solara run app.py
"""

import json

import solara
from mesa.visualization import SolaraViz, make_plot_component

from labour_sim.config import SimConfig, _PRESET_DIR, _deep_merge, list_presets
from labour_sim.sim.model import LabourMarketModel
from labour_sim.viz.agent_space import AgentSpacePanel
from labour_sim.viz.diagnostics import BeveridgePanel, FlowMatrixPanel, InequalityPanel


def build_config(
    preset: str = "baseline",
    seed: int = 42,
    n_workers: int = 2000,
    horizon_years: int = 20,
    capability_growth: float = 0.035,
    ai_cost_decline: float = 0.015,
    imitation_weight: float = 0.35,
    benefit_level: float = 0.45,
    retraining_subsidy: float = 0.3,
) -> SimConfig:
    """Sliders set the base; a named preset's overrides are pinned on top
    (choose 'baseline' for full slider control)."""
    base = SimConfig().model_dump()
    base.update(seed=int(seed), n_workers=n_workers, horizon_years=horizon_years)
    base["capability"] = {**base["capability"], "growth_rate": capability_growth}
    base["ai_cost"] = {**base["ai_cost"], "decline_rate": ai_cost_decline}
    base["adoption"] = {**base["adoption"], "imitation_weight": imitation_weight}
    base["policy"] = {
        **base["policy"],
        "benefit_level": benefit_level,
        "retraining_subsidy": retraining_subsidy,
    }
    payload = json.loads((_PRESET_DIR / f"{preset}.json").read_text(encoding="utf-8"))
    return SimConfig.model_validate(_deep_merge(base, payload.get("overrides", {})))


class DashboardModel(LabourMarketModel):
    """Adapter exposing flat kwargs so SolaraViz sliders can rebuild the model."""

    def __init__(
        self,
        preset: str = "baseline",
        seed: int = 42,
        n_workers: int = 2000,
        capability_growth: float = 0.035,
        ai_cost_decline: float = 0.015,
        imitation_weight: float = 0.35,
        benefit_level: float = 0.45,
        retraining_subsidy: float = 0.3,
    ) -> None:
        super().__init__(
            build_config(
                preset=preset,
                seed=seed,
                n_workers=n_workers,
                capability_growth=capability_growth,
                ai_cost_decline=ai_cost_decline,
                imitation_weight=imitation_weight,
                benefit_level=benefit_level,
                retraining_subsidy=retraining_subsidy,
            )
        )


MODEL_PARAMS = {
    "preset": {
        "type": "Select",
        "value": "baseline",
        "values": list_presets(),
        "label": "Preset scenario (pins its own parameters)",
    },
    "seed": {"type": "InputText", "value": 42, "label": "Random seed"},
    "n_workers": {
        "type": "SliderInt",
        "value": 2000,
        "min": 500,
        "max": 10000,
        "step": 500,
        "label": "Workers",
    },
    "capability_growth": {
        "type": "SliderFloat",
        "value": 0.035,
        "min": 0.0,
        "max": 0.12,
        "step": 0.005,
        "label": "AI capability growth /mo",
    },
    "ai_cost_decline": {
        "type": "SliderFloat",
        "value": 0.015,
        "min": 0.0,
        "max": 0.05,
        "step": 0.005,
        "label": "AI cost decline /mo",
    },
    "imitation_weight": {
        "type": "SliderFloat",
        "value": 0.35,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "label": "Adoption imitation",
    },
    "benefit_level": {
        "type": "SliderFloat",
        "value": 0.45,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "label": "Benefit level",
    },
    "retraining_subsidy": {
        "type": "SliderFloat",
        "value": 0.3,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "label": "Retraining subsidy",
    },
}


@solara.component
def ExportPanel(model) -> None:
    def to_csv() -> str:
        return model.datacollector.get_model_vars_dataframe().to_csv(index_label="tick")

    with solara.Card("Export"):
        solara.FileDownload(to_csv, filename="labour_sim_run.csv", label="Download metrics CSV")


LabourMarketPlot = make_plot_component(
    {"unemployment_rate": "tab:red", "discouraged_share": "tab:orange"}
)
AdoptionPlot = make_plot_component(
    {"adoption_share": "tab:blue", "automated_task_share": "tab:purple", "capability_level": "tab:gray"}
)
OutputPlot = make_plot_component({"output_index": "tab:green", "mean_wage": "tab:olive"})
InequalityPlot = make_plot_component({"wage_gini": "tab:brown"})


def make_dashboard() -> SolaraViz:
    model = DashboardModel()
    return SolaraViz(
        model,
        components=[
            AgentSpacePanel,
            LabourMarketPlot,
            AdoptionPlot,
            OutputPlot,
            InequalityPlot,
            InequalityPanel,
            BeveridgePanel,
            FlowMatrixPanel,
            ExportPanel,
        ],
        model_params=MODEL_PARAMS,
        name="AI Labour Market Simulator (ABM)",
    )
