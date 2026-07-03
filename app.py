"""Solara entry point. Run with: uv run solara run app.py

Two pages: the live SolaraViz dashboard (agent view + time series) and the
research page (Monte Carlo bands, scenario files, parameter sweeps).
"""

import solara

from labour_sim.viz.dashboard import make_dashboard
from labour_sim.viz.research_page import ResearchPage


@solara.component
def LivePage() -> None:
    viz = solara.use_memo(make_dashboard, [])
    solara.display(viz)


routes = [
    solara.Route(path="/", component=LivePage, label="Live model"),
    solara.Route(path="research", component=ResearchPage, label="Research"),
]
