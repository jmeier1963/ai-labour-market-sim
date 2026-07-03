"""Generate all result figures and tables for the paper.

Run from the project root:  uv run python paper/make_figures.py
Parallel Monte Carlo is safe here because this script guards __main__.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from labour_sim.batch import monte_carlo, set_by_path, sweep_2d
from labour_sim.config import SimConfig, load_preset
from labour_sim.sim.model import LabourMarketModel

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
DPI = 220
PROCESSES = 6

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": DPI,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

COLORS = {"baseline": "#0072B2", "fast_takeoff": "#D55E00", "high_friction": "#009E73"}
LABELS = {"baseline": "Baseline", "fast_takeoff": "Fast takeoff", "high_friction": "High friction"}


def run_preset(name: str, seed: int = 42, n_workers: int = 2000):
    cfg = load_preset(name).model_copy(update={"seed": seed, "n_workers": n_workers})
    model = LabourMarketModel(cfg)
    model.run()
    return model.datacollector.get_model_vars_dataframe(), model


def no_ai_config(seed: int = 42, n_workers: int = 2000) -> SimConfig:
    return SimConfig(
        seed=seed,
        n_workers=n_workers,
        capability={"initial": 0.0, "ceiling": 0.0, "growth_rate": 0.0, "noise_sigma": 0.0},
    )


def fig_schematic() -> None:
    """Graphical abstract: data -> agents -> tick cycle -> emergent outcomes."""
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.6)

    def box(x, y, w, h, title, lines, face, title_size=9):
        ax.add_patch(
            plt.Rectangle((x, y), w, h, facecolor=face, edgecolor="#333", lw=0.8, zorder=1)
        )
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", fontsize=title_size, weight="bold")
        for i, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.62 - 0.30 * i, line, ha="center", fontsize=7.4)

    def arrow(x0, y0, x1, y1):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "-|>", "color": "#333", "lw": 1.4},
            zorder=3,
        )

    box(
        0.15,
        1.3,
        2.1,
        2.0,
        "Empirical grounding",
        ["24 tasks (E0/E1/E2)", "27 occupations", "(incl. junior/senior splits)", "8 sectors", "exposure, difficulty, augmentation"],
        "#E8F0FE",
    )
    box(
        2.75,
        2.5,
        2.3,
        1.75,
        "Firms (~400)",
        ["task-level AI adoption", "ROI + imitation + hurdle", "vacancies, prices", "entry / exit (vintage)"],
        "#FDE9D9",
    )
    box(
        2.75,
        0.35,
        2.3,
        1.75,
        "Workers (5,000)",
        ["search & applications", "reservation wages", "retraining in task space", "discouragement"],
        "#E2F0D9",
    )
    box(
        5.55,
        1.3,
        2.2,
        2.0,
        "Monthly interaction",
        ["capability frontier ↑", "AI price ↓", "product-market rebound", "decentralized matching", "wage bargaining"],
        "#FFF2CC",
    )
    box(
        8.25,
        1.3,
        2.1,
        2.0,
        "Emergent outcomes",
        ["adoption S-curves", "displacement waves", "wage collapse & scarring", "Beveridge loops", "ambiguous policy effects"],
        "#F4E1F7",
    )
    arrow(2.25, 2.3, 2.75, 2.9)
    arrow(2.25, 2.3, 2.75, 1.6)
    arrow(5.05, 3.1, 5.55, 2.75)
    arrow(5.05, 1.2, 5.55, 1.75)
    arrow(7.75, 2.3, 8.25, 2.3)
    ax.text(
        5.25,
        4.4,
        "Agent-based model of AI adoption in the labour market — nothing is a sigmoid of time",
        ha="center",
        fontsize=10,
        style="italic",
    )
    fig.savefig(FIG_DIR / "fig_schematic.png")
    plt.close(fig)
    print("schematic done")


def fig_scenarios() -> None:
    frames = {}
    for name in ("baseline", "fast_takeoff", "high_friction"):
        frames[name], _ = run_preset(name)
        print(f"scenario {name} done")

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2))
    for name, frame in frames.items():
        color, label = COLORS[name], LABELS[name]
        x = frame.index / 12.0
        axes[0, 0].plot(x, frame["capability_level"], color=color, label=label)
        axes[0, 0].plot(x, frame["ai_price"], color=color, ls="--", alpha=0.6)
        axes[0, 1].plot(x, frame["adoption_share"], color=color, label=label)
        axes[1, 0].plot(x, frame["unemployment_rate"] + frame["discouraged_share"], color=color)
        axes[1, 1].plot(x, frame["output_index"], color=color)

    axes[0, 0].set_title("AI capability (solid) and AI price (dashed)")
    axes[0, 0].set_ylabel("index")
    axes[0, 1].set_title("Firm task-adoption share (emergent S-curve)")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].set_title("Labour-market distress (unemployed + discouraged)")
    axes[1, 0].set_ylabel("share of workforce")
    axes[1, 1].set_title("Output volume index")
    for ax in axes.flat:
        ax.set_xlabel("years")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_scenarios.png")
    plt.close(fig)

    frames["fast_takeoff"].to_csv(FIG_DIR / "_fast_run.csv")


def fig_montecarlo() -> None:
    cfg = load_preset("fast_takeoff").model_copy(update={"seed": 42, "n_workers": 1500})
    result = monte_carlo(cfg, runs=24, processes=PROCESSES)
    print("monte carlo done")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ax, metric, title in (
        (axes[0], "unemployment_rate", "Unemployment rate"),
        (axes[1], "adoption_share", "Firm task-adoption share"),
    ):
        bands = result.bands(metric)
        x = bands.index / 12.0
        ax.fill_between(x, bands["p5"], bands["p95"], alpha=0.18, color="#D55E00", label="p5–p95")
        ax.fill_between(x, bands["p25"], bands["p75"], alpha=0.35, color="#D55E00", label="p25–p75")
        ax.plot(x, bands["p50"], color="#8B3A00", label="median")
        ax.set_title(f"{title} (24 seeds, fast takeoff)")
        ax.set_xlabel("years")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_montecarlo.png")
    plt.close(fig)


def fig_inequality_beveridge() -> None:
    import pandas as pd

    fast = pd.read_csv(FIG_DIR / "_fast_run.csv", index_col=0)
    base_cfg = no_ai_config()
    base_model = LabourMarketModel(base_cfg)
    base_model.run()
    base = base_model.datacollector.get_model_vars_dataframe()
    print("no-AI reference done")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    x = fast.index / 12.0
    ax = axes[0]
    ax.fill_between(x, fast["wage_p10"], fast["wage_p90"], alpha=0.15, color="#0072B2")
    ax.fill_between(x, fast["wage_p25"], fast["wage_p75"], alpha=0.3, color="#0072B2")
    ax.plot(x, fast["wage_p50"], color="#0072B2", label="median wage")
    ax2 = ax.twinx()
    ax2.plot(x, fast["wage_gini"], color="#D55E00", ls=":", label="Gini (right)")
    ax2.set_ylabel("wage Gini", color="#D55E00")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.set_title("Wage distribution under fast takeoff (p10–p90 fan)")
    ax.set_xlabel("years")
    ax.set_ylabel("wage (median t0 = 1)")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    for frame, cmap, label in ((base, "Blues", "no-AI baseline"), (fast, "Oranges", "fast takeoff")):
        u = frame["unemployment_rate"]
        v = frame["vacancy_count"] / 2000.0
        ax.scatter(u, v, c=frame.index, cmap=cmap, s=7, label=label)
    ax.set_title("Beveridge curve trace (colour = time)")
    ax.set_xlabel("unemployment rate")
    ax.set_ylabel("vacancy rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_inequality_beveridge.png")
    plt.close(fig)


def fig_policy() -> None:
    fast = load_preset("fast_takeoff").model_dump()
    experiments = {
        "Benefit level\n0.25 vs 0.65": ("policy.benefit_level", 0.25, 0.65),
        "Firing cost\n0.05 vs 0.60": ("policy.firing_cost", 0.05, 0.60),
        "Matching efficiency\n0.25 vs 0.65": ("matching.efficiency", 0.25, 0.65),
    }
    seeds = [1, 2, 3, 4, 5]

    def peaks(path: str, value: float) -> list[float]:
        out = []
        for seed in seeds:
            cfg = SimConfig.model_validate(
                {**fast, "seed": seed, "n_workers": 1200, "horizon_years": 12}
            )
            cfg = set_by_path(cfg, path, value)
            frames = monte_carlo(cfg, runs=1, processes=1).runs
            frame = frames[0]
            out.append(float((frame["unemployment_rate"] + frame["discouraged_share"]).max()))
        return out

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    positions, ticks = [], []
    results_json = {}
    for i, (label, (path, low, high)) in enumerate(experiments.items()):
        for j, value in enumerate((low, high)):
            data = peaks(path, value)
            results_json[f"{path}={value}"] = data
            pos = i * 3 + j
            color = "#0072B2" if j == 0 else "#D55E00"
            ax.boxplot(
                [data],
                positions=[pos],
                widths=0.7,
                patch_artist=True,
                boxprops={"facecolor": color, "alpha": 0.5},
                medianprops={"color": "black"},
            )
            print(f"policy {path}={value} done")
        positions.append(i * 3 + 0.5)
        ticks.append(label)
    ax.set_xticks(positions, ticks)
    ax.set_ylabel("peak distress (u + discouraged)")
    ax.set_title("Policy and friction experiments under fast takeoff (5 seeds; blue = low value, orange = high)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_policy.png")
    plt.close(fig)
    (FIG_DIR / "_policy.json").write_text(json.dumps(results_json, indent=2))


def fig_sweep() -> None:
    cfg = load_preset("fast_takeoff").model_copy(
        update={"seed": 7, "n_workers": 800, "horizon_years": 12}
    )
    growth = [0.02, 0.05, 0.08, 0.11]
    efficiency = [0.25, 0.4, 0.55, 0.7]
    grid = sweep_2d(
        cfg,
        "capability.growth_rate",
        growth,
        "matching.efficiency",
        efficiency,
        stat="peak_distress",
        runs_per_point=2,
        processes=PROCESSES,
    )
    print("sweep done")

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    image = ax.imshow(grid.to_numpy(), cmap="RdYlGn_r", aspect="auto", origin="lower")
    ax.set_xticks(range(len(growth)), [f"{g:.2f}" for g in growth])
    ax.set_yticks(range(len(efficiency)), [f"{e:.2f}" for e in efficiency])
    ax.set_xlabel("AI capability growth rate (per month)")
    ax.set_ylabel("matching efficiency")
    for i in range(len(efficiency)):
        for j in range(len(growth)):
            ax.text(j, i, f"{grid.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="peak distress (median of 2 seeds)")
    ax.set_title("Peak distress: AI speed × reallocation friction")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_sweep.png")
    plt.close(fig)


def fig_dependencies() -> None:
    """Static force-directed dependency graph for the paper appendix."""
    import networkx as nx

    from dependency_data import EDGES, GROUPS, NODES, POSITIONS

    graph = nx.DiGraph()
    for node_id, group, label in NODES:
        graph.add_node(node_id, group=group, label=label)
    graph.add_edges_from(EDGES)

    pos = POSITIONS

    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    ax.axis("off")
    degree = dict(graph.degree())
    sizes = [120 + 55 * degree[n] for n in graph.nodes]
    colors = [GROUPS[graph.nodes[n]["group"]]["color"] for n in graph.nodes]

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=9,
        edge_color="#888780",
        alpha=0.45,
        width=0.9,
        node_size=sizes,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_size=sizes, node_color=colors, edgecolors="#444", linewidths=0.6
    )
    labels = {n: graph.nodes[n]["label"] for n in graph.nodes}
    label_pos = {n: (x, y - 0.26) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(graph, label_pos, labels=labels, ax=ax, font_size=7.5)

    handles = [
        plt.Line2D(
            [], [], marker="o", ls="", markersize=8, color=spec["color"], label=spec["name"]
        )
        for spec in GROUPS.values()
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    ax.margins(0.05)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_dependencies.png")
    plt.close(fig)
    print("dependency graph done")


def fig_seniority() -> None:
    """Phase A seniority split: pyramid inversion in knowledge occupations."""
    fast_cfg = load_preset("fast_takeoff").model_copy(update={"seed": 42, "n_workers": 2000})
    fast_model = LabourMarketModel(fast_cfg)
    fast_model.run()
    fast = fast_model.datacollector.get_model_vars_dataframe()

    base_model = LabourMarketModel(no_ai_config())
    base_model.run()
    base = base_model.datacollector.get_model_vars_dataframe()
    print("seniority scenario runs done")

    def group_wage_median(model, suffix: str) -> float:
        wages = [
            w.wage
            for w in model.workers
            if w.state == "employed" and w.occupation.endswith(suffix)
        ]
        return float(np.median(wages)) if wages else float("nan")

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    x = fast.index / 12.0

    ax = axes[0]
    ax.plot(x, fast["knowledge_junior_share"], color="#D55E00", label="junior (fast takeoff)")
    ax.plot(x, fast["knowledge_senior_share"], color="#0072B2", label="senior (fast takeoff)")
    ax.plot(x, base["knowledge_junior_share"], color="#D55E00", ls=":", alpha=0.7, label="junior (no AI)")
    ax.plot(x, base["knowledge_senior_share"], color="#0072B2", ls=":", alpha=0.7, label="senior (no AI)")
    ax.set_title("Knowledge-occupation employment (share of workforce)")
    ax.set_xlabel("years")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    ax.plot(x, fast["knowledge_pyramid_ratio"], color="#D55E00", label="fast takeoff")
    ax.plot(x, base["knowledge_pyramid_ratio"], color="#0072B2", ls=":", label="no AI")
    ax.axhline(1.0, color="#666", lw=0.8, ls="--")
    ax.annotate("inversion (ratio = 1)", (11, 1.08), fontsize=7, color="#666")
    ax.set_title("Pyramid ratio: employed juniors per senior")
    ax.set_xlabel("years")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[2]
    window = slice(6, 96)
    ax.plot(x[window], fast["entrant_junior_share"].iloc[window], color="#009E73", label="entrants (staff)")
    ax.plot(x[window], fast["incumbent_junior_share"].iloc[window], color="#CC79A7", label="t0 incumbents (staff)")
    ax.set_title("Junior share of split-occupation staff")
    ax.set_xlabel("years")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_seniority.png")
    plt.close(fig)

    # Multi-seed medians for the paper text.
    ratios_start, ratios_end, junior_drop, senior_change, ent, inc = [], [], [], [], [], []
    premium_start, premium_end = [], []
    for seed in (1, 2, 3, 4, 5):
        cfg = load_preset("fast_takeoff").model_copy(update={"seed": seed, "n_workers": 1500})
        model = LabourMarketModel(cfg)
        model.run(12)
        premium_start.append(group_wage_median(model, "-SR") / group_wage_median(model, "-JR"))
        model.run(228)
        premium_end.append(group_wage_median(model, "-SR") / group_wage_median(model, "-JR"))
        frame = model.datacollector.get_model_vars_dataframe()
        ratios_start.append(frame["knowledge_pyramid_ratio"].iloc[:12].mean())
        ratios_end.append(frame["knowledge_pyramid_ratio"].iloc[-12:].mean())
        junior_drop.append(
            frame["knowledge_junior_share"].iloc[-12:].mean()
            / frame["knowledge_junior_share"].iloc[:12].mean()
        )
        senior_change.append(
            frame["knowledge_senior_share"].iloc[-12:].mean()
            / frame["knowledge_senior_share"].iloc[:12].mean()
        )
        w = frame.iloc[18:48]
        ent.append(w["entrant_junior_share"].mean())
        inc.append(w["incumbent_junior_share"].mean())
        print(f"seniority seed {seed} done")

    stats = {
        "pyramid_ratio_start_median": float(np.median(ratios_start)),
        "pyramid_ratio_end_median": float(np.median(ratios_end)),
        "junior_employment_end_over_start_median": float(np.median(junior_drop)),
        "senior_employment_end_over_start_median": float(np.median(senior_change)),
        "entrant_junior_share_window_median": float(np.median(ent)),
        "incumbent_junior_share_window_median": float(np.median(inc)),
        "senior_junior_wage_premium_y1_median": float(np.median(premium_start)),
        "senior_junior_wage_premium_end_median": float(np.median(premium_end)),
    }
    (FIG_DIR / "_seniority.json").write_text(json.dumps(stats, indent=2))
    print("seniority stats:", stats)


def calibration_table() -> None:
    model = LabourMarketModel(no_ai_config(seed=42))
    model.run()
    frame = model.datacollector.get_model_vars_dataframe().iloc[24:]
    stats = {
        "unemployment_rate": float(frame["unemployment_rate"].mean()),
        "job_finding_rate": float(frame["job_finding_rate"].mean()),
        "separation_rate": float(frame["separation_rate"].mean()),
        "wage_gini": float(frame["wage_gini"].iloc[-1]),
        "employment_drift": float(
            frame["employment_rate"].iloc[-1] - frame["employment_rate"].iloc[0]
        ),
    }
    (FIG_DIR / "_calibration.json").write_text(json.dumps(stats, indent=2))
    print("calibration table done:", stats)


if __name__ == "__main__":
    fig_schematic()
    calibration_table()
    fig_scenarios()
    fig_inequality_beveridge()
    fig_montecarlo()
    fig_policy()
    fig_sweep()
    fig_seniority()
    fig_dependencies()
    print("ALL FIGURES DONE")
