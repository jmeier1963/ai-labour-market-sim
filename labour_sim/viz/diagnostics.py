"""Research diagnostics panels: wage-percentile fan (inequality), cumulative
occupation flow matrix (reallocation), and Beveridge curve (tipping)."""

import matplotlib.pyplot as plt
import numpy as np
import solara
from mesa.visualization.utils import update_counter


@solara.component
def InequalityPanel(model) -> None:
    update_counter.get()
    frame = model.datacollector.get_model_vars_dataframe()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    if len(frame) > 1:
        x = frame.index
        ax.fill_between(x, frame["wage_p10"], frame["wage_p90"], alpha=0.15, color="tab:blue")
        ax.fill_between(x, frame["wage_p25"], frame["wage_p75"], alpha=0.3, color="tab:blue")
        ax.plot(x, frame["wage_p50"], color="tab:blue", label="median wage")
        ax.legend(fontsize=7)
    ax.set_title("Wage distribution fan (p10-p90)", fontsize=9)
    ax.set_xlabel("month", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    solara.FigureMatplotlib(fig)
    plt.close(fig)


@solara.component
def FlowMatrixPanel(model) -> None:
    update_counter.get()
    occupations = list(model.dataset.occupations)
    n = len(occupations)
    matrix = np.zeros((n, n))
    index = {occ: i for i, occ in enumerate(occupations)}
    for (source, target), count in model.occupation_flows.items():
        matrix[index[source], index[target]] = count
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto")
    labels = [occ.replace("SOC-", "") for occ in occupations]
    ax.set_xticks(range(n), labels, fontsize=6, rotation=90)
    ax.set_yticks(range(n), labels, fontsize=6)
    ax.set_xlabel("to occupation (SOC)", fontsize=8)
    ax.set_ylabel("from occupation (SOC)", fontsize=8)
    ax.set_title("Cumulative occupation switches at hire", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.04)
    fig.tight_layout()
    solara.FigureMatplotlib(fig)
    plt.close(fig)


@solara.component
def BeveridgePanel(model) -> None:
    update_counter.get()
    frame = model.datacollector.get_model_vars_dataframe()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    if len(frame) > 1:
        u = frame["unemployment_rate"]
        v = frame["vacancy_count"] / max(1, len(model.workers))
        points = ax.scatter(u, v, c=frame.index, cmap="viridis", s=8)
        fig.colorbar(points, ax=ax, fraction=0.04, label="month")
    ax.set_xlabel("unemployment rate", fontsize=8)
    ax.set_ylabel("vacancy rate", fontsize=8)
    ax.set_title("Beveridge curve trace (outward shift = reallocation stress)", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    solara.FigureMatplotlib(fig)
    plt.close(fig)
