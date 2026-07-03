"""Agent-space view: workers as dots (colored by state) clustered around their
employer; firms as squares (sized by headcount, colored by AI adoption share);
the unemployed drift into a central ring, discouraged workers into the core.

Sector clusters are arranged on a circle. Positions are deterministic per agent
so the view is stable across re-renders and reproducible per seed.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import solara
from matplotlib import colormaps, cm, colors as mcolors
from mesa.visualization.utils import update_counter

from labour_sim.sim.workers import DISCOURAGED, EMPLOYED, SEARCHING

SECTOR_RADIUS = 4.2
FIRM_SPREAD = 1.35
WORKER_SPREAD = 0.30
UNEMPLOYED_RING = (0.9, 1.6)
DISCOURAGED_RING = (0.0, 0.55)

STATE_COLORS = {EMPLOYED: "#2a9d8f", SEARCHING: "#e76f51", DISCOURAGED: "#6c757d"}
ADOPTION_CMAP = colormaps["plasma"]


def _sector_centers(model) -> dict[str, tuple[float, float]]:
    sector_ids = list(model.dataset.sectors)
    centers = {}
    for i, sector_id in enumerate(sector_ids):
        angle = 2.0 * math.pi * i / len(sector_ids)
        centers[sector_id] = (SECTOR_RADIUS * math.cos(angle), SECTOR_RADIUS * math.sin(angle))
    return centers


def _jitter(uid: int, salt: int, spread: float) -> tuple[float, float]:
    rng = np.random.default_rng(uid * 2654435761 % 2**32 + salt)
    radius = spread * math.sqrt(rng.uniform())
    angle = 2.0 * math.pi * rng.uniform()
    return radius * math.cos(angle), radius * math.sin(angle)


def _firm_position(firm, centers) -> tuple[float, float]:
    if not hasattr(firm, "_viz_pos"):
        cx, cy = centers[firm.sector]
        dx, dy = _jitter(firm.unique_id, salt=1, spread=FIRM_SPREAD)
        firm._viz_pos = (cx + dx, cy + dy)
    return firm._viz_pos


def _worker_position(worker, centers) -> tuple[float, float]:
    if worker.state == EMPLOYED and worker.employer is not None:
        fx, fy = _firm_position(worker.employer, centers)
        dx, dy = _jitter(worker.unique_id, salt=2, spread=WORKER_SPREAD)
        return fx + dx, fy + dy
    lo, hi = UNEMPLOYED_RING if worker.state == SEARCHING else DISCOURAGED_RING
    rng = np.random.default_rng(worker.unique_id * 2654435761 % 2**32 + 3)
    radius = lo + (hi - lo) * rng.uniform()
    angle = 2.0 * math.pi * rng.uniform()
    return radius * math.cos(angle), radius * math.sin(angle)


@solara.component
def AgentSpacePanel(model) -> None:
    update_counter.get()

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    centers = _sector_centers(model)

    xs, ys, sizes, face = [], [], [], []
    for firm in model.firms:
        x, y = _firm_position(firm, centers)
        xs.append(x)
        ys.append(y)
        sizes.append(14.0 + 3.0 * math.sqrt(max(1, firm.headcount)))
        face.append(ADOPTION_CMAP(firm.adoption_share()))
    ax.scatter(xs, ys, s=sizes, c=face, marker="s", edgecolors="black", linewidths=0.3, zorder=2)

    for state, color in STATE_COLORS.items():
        wx, wy = [], []
        for worker in model.workers:
            if worker.state != state:
                continue
            x, y = _worker_position(worker, centers)
            wx.append(x)
            wy.append(y)
        ax.scatter(wx, wy, s=2.5, c=color, alpha=0.55, linewidths=0, zorder=1, label=state)

    for sector_id, (cx, cy) in centers.items():
        ax.annotate(
            model.dataset.sectors[sector_id].name,
            (cx, cy + FIRM_SPREAD + 0.35),
            ha="center",
            fontsize=7,
            color="#333",
        )

    ax.set_title(
        f"Month {model.steps} | capability {model.capability.level:.2f} | "
        f"AI price {model.ai_price:.2f}",
        fontsize=9,
    )
    ax.legend(loc="lower right", fontsize=7, markerscale=3)
    fig.colorbar(
        cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=ADOPTION_CMAP),
        ax=ax,
        fraction=0.04,
        label="firm AI adoption share",
    )
    ax.set_xlim(-6.6, 6.6)
    ax.set_ylim(-6.6, 6.6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()

    solara.FigureMatplotlib(fig)
    plt.close(fig)
