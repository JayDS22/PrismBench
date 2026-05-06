"""
Utility functions and convenience wrappers for the PrismBench framework.

Thin facade over the implementation modules under `src/`. Notebooks and
external callers should import from here rather than reaching into the
internal package layout, so the public surface stays stable while the
internals evolve.

Import as:

    import prismbench_utils as pbu
"""

import json
import logging
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

# Re-export the router public API so a single import covers MCDM use cases.
from src.router import (
    PRESETS,
    build_routing_matrix,
    find_top1_breakpoints,
    kendall_tau_table,
    load_summary,
    normalize,
    pareto_2d,
    pareto_optimal,
    plot_pareto_2d,
    plot_sensitivity,
    promethee,
    recommend,
    sensitivity_sweep,
    topsis,
    wsm,
)
from src.task_runner import (
    load_agents_config,
    load_tasks_config,
    run_batch,
    run_single,
)
from src.utils import RESULTS_DIR

_LOG = logging.getLogger(__name__)


# #############################################################################
# Constants
# #############################################################################

DIMENSIONS = (
    "primary",
    "D2_combined",
    "D3_combined",
    "D4_time_sec",
    "D5_cost_usd",
    "D5_carbon_kg",
    "D6_cv",
)

LOWER_IS_BETTER = {"D4_time_sec", "D5_cost_usd", "D5_carbon_kg", "D6_cv"}

MAIN_TASKS = ("HD-PRED-01", "TAXI-PRED-01", "AR-PRED-01")


# #############################################################################
# Scorecard convenience
# #############################################################################


def load_master_scorecard(detailed: bool = False) -> pd.DataFrame:
    """
    Load the per-run or per-(agent, task) master scorecard.

    :param detailed: if True, return one row per run; otherwise return the
        per-(agent, task) summary that the router consumes.
    :return: scorecard DataFrame.
    """
    name = "master_scorecard.csv" if detailed else "master_scorecard_summary.csv"
    path = Path(RESULTS_DIR) / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.scorecard` after a benchmark."
        )
    return pd.read_csv(path)


def load_run_scorecard(agent: str, task_id: str, run_id: int) -> dict:
    """
    Load the per-run `scorecard.json` for one (agent, task, run) cell.
    """
    path = Path(RESULTS_DIR) / agent / task_id / f"run_{run_id}" / "scorecard.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return json.loads(path.read_text())


# #############################################################################
# Routing convenience
# #############################################################################


def top_agent(weights: Mapping[str, float], method: str = "wsm") -> str:
    """
    Return the single top-ranked agent under the given weight vector.

    :param weights: dimension -> weight mapping (must sum to 1.0).
    :param method: 'wsm', 'topsis', or 'promethee'.
    :return: agent_id of the top-ranked agent.
    """
    rec = recommend(weights, methods=(method,))
    series = rec["rankings"][method]
    return str(series.index[0])


def compare_presets(
    presets: Iterable[str] = ("balanced", "accuracy", "frugal", "green"),
    method: str = "wsm",
) -> pd.DataFrame:
    """
    Tabulate the top-3 agents under each preset for a quick side-by-side.
    """
    rows = {}
    for name in presets:
        if name not in PRESETS:
            raise KeyError(f"unknown preset {name!r}; known: {list(PRESETS)}")
        rec = recommend(PRESETS[name], methods=(method,))
        rows[name] = list(rec["rankings"][method].index[:3])
    return pd.DataFrame(rows, index=["#1", "#2", "#3"]).T


# #############################################################################
# Pareto convenience
# #############################################################################


def pareto_corner(matrix: pd.DataFrame, x_col: str, y_col: str) -> list[str]:
    """
    Pareto-optimal agents in a 2D projection, with directions inferred from
    the lower-is-better convention in `LOWER_IS_BETTER`.
    """
    mask = pareto_2d(
        matrix,
        x_col=x_col,
        y_col=y_col,
        x_lower_better=x_col in LOWER_IS_BETTER,
        y_lower_better=y_col in LOWER_IS_BETTER,
    )
    return list(mask[mask].index)


__all__ = [
    "DIMENSIONS",
    "LOWER_IS_BETTER",
    "MAIN_TASKS",
    "PRESETS",
    "build_routing_matrix",
    "compare_presets",
    "find_top1_breakpoints",
    "kendall_tau_table",
    "load_agents_config",
    "load_master_scorecard",
    "load_run_scorecard",
    "load_summary",
    "load_tasks_config",
    "normalize",
    "pareto_2d",
    "pareto_corner",
    "pareto_optimal",
    "plot_pareto_2d",
    "plot_sensitivity",
    "promethee",
    "recommend",
    "run_batch",
    "run_single",
    "sensitivity_sweep",
    "top_agent",
    "topsis",
    "wsm",
]
