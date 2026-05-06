"""
Router: preference-aware agent recommendation via MCDM.

Methods:
- WSM (Weighted Sum Model)
- TOPSIS (distance from ideal / anti-ideal)
- PROMETHEE-II (pairwise preference net flows)

Input  : results/master_scorecard_summary.csv (one row per (agent, task)).
Output : ranked agent list per method, Kendall's τ between methods,
         and the Pareto-optimal subset.

Usage:
    python -m src.router --preset balanced
    python -m src.router --preset frugal --all-methods
    python -m src.router --weights "D1=0.4,D2=0.1,D3=0.1,D4_speed=0.1,D5_cost=0.15,D5_carbon=0.1,D6_robust=0.05"
"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import RESULTS_DIR, PROJECT_ROOT, log

# Plots intended for the writeup live in `figures/` (tracked). Per-run
# scorecards stay under `results/` (gitignored).
FIGURES_DIR = PROJECT_ROOT / "figures"


# ============================================================================
# Dimensions: (display_name, df_column, direction) where direction is +1 if
# higher is better, -1 if lower is better.
# ============================================================================
DIMENSIONS = [
    ("D1",        "primary",       +1),
    ("D2",        "D2_combined",   +1),
    ("D3",        "D3_combined",   +1),
    ("D4_speed",  "D4_time_sec",   -1),
    ("D5_cost",   "D5_cost_usd",   -1),
    ("D5_carbon", "D5_carbon_kg",  -1),
    ("D6_robust", "D6_cv",         -1),
]
DIM_NAMES = [d[0] for d in DIMENSIONS]


# Main benchmark tasks used for routing. Diagnostic results (e.g. HD-DESC-01
# from the pandasai mismatch probe, HD-PRED-01_diag10 from the smolagents
# step-cap probe) are excluded so coverage counts reflect the canonical
# benchmark, not one-off experiments.
MAIN_TASKS = ["HD-PRED-01", "TAXI-PRED-01", "AR-PRED-01"]


# ============================================================================
# Preset weight profiles. Must sum to 1.0.
# ============================================================================
PRESETS = {
    "balanced":   {n: 1 / len(DIM_NAMES) for n in DIM_NAMES},
    "accuracy":   {"D1": 0.60, "D2": 0.10, "D3": 0.10, "D4_speed": 0.05, "D5_cost": 0.05, "D5_carbon": 0.05, "D6_robust": 0.05},
    "frugal":     {"D1": 0.20, "D2": 0.10, "D3": 0.10, "D4_speed": 0.05, "D5_cost": 0.30, "D5_carbon": 0.20, "D6_robust": 0.05},
    "quality":    {"D1": 0.30, "D2": 0.25, "D3": 0.25, "D4_speed": 0.05, "D5_cost": 0.05, "D5_carbon": 0.05, "D6_robust": 0.05},
    "green":      {"D1": 0.20, "D2": 0.05, "D3": 0.05, "D4_speed": 0.05, "D5_cost": 0.10, "D5_carbon": 0.45, "D6_robust": 0.10},
    "production": {"D1": 0.30, "D2": 0.15, "D3": 0.10, "D4_speed": 0.10, "D5_cost": 0.10, "D5_carbon": 0.05, "D6_robust": 0.20},
}


# ============================================================================
# Data preparation
# ============================================================================

def load_summary(path=None):
    p = Path(path) if path else (RESULTS_DIR / "master_scorecard_summary.csv")
    if not p.exists():
        raise FileNotFoundError(f"{p} not found, run `python -m src.scorecard` first")
    return pd.read_csv(p)


def build_routing_matrix(summary_df, tasks=None):
    """Aggregate per-(agent, task) rows into one row per agent with the 7
    routing dimensions populated, plus coverage metadata.

    D1 (primary) is per-task min-max normalized first (F1 for classification
    or NLP, 1/(1+RMSE) for regression so higher is better), then averaged
    across tasks. Other dimensions are mean across the agent's tasks.

    Args:
        summary_df : the master_scorecard_summary.csv frame.
        tasks      : list of task_ids to include. Defaults to MAIN_TASKS so
                     diagnostic experiments don't pollute coverage counts.

    Output also includes:
      n_tasks_total      : count of tasks considered (denominator)
      n_tasks_successful : tasks where the agent produced a valid primary metric
    """
    if tasks is None:
        tasks = MAIN_TASKS
    df = summary_df[summary_df.task_id.isin(tasks)].copy()
    n_tasks_total = len(tasks)

    primary_pieces = []
    for task in df.task_id.unique():
        sub = df[df.task_id == task].copy()
        sub["primary_raw"] = sub["D1_f1"]
        is_regression = sub["primary_raw"].isna() & sub["D1_rmse"].notna()
        sub.loc[is_regression, "primary_raw"] = 1.0 / (1.0 + sub.loc[is_regression, "D1_rmse"])
        v = sub["primary_raw"]
        lo, hi = v.min(skipna=True), v.max(skipna=True)
        if pd.notna(lo) and pd.notna(hi) and hi > lo:
            sub["primary_norm"] = (v - lo) / (hi - lo)
        else:
            sub["primary_norm"] = np.nan
        primary_pieces.append(sub[["agent", "primary_norm"]])

    primary_long = pd.concat(primary_pieces)
    primary = primary_long.groupby("agent")["primary_norm"].mean().reset_index()
    primary = primary.rename(columns={"primary_norm": "primary"})

    coverage = (primary_long.groupby("agent")["primary_norm"]
                            .apply(lambda s: int(s.notna().sum()))
                            .reset_index()
                            .rename(columns={"primary_norm": "n_tasks_successful"}))

    other_cols = ["D2_combined", "D3_combined", "D4_time_sec",
                  "D5_cost_usd", "D5_carbon_kg", "D6_cv"]
    other = df.groupby("agent")[other_cols].mean().reset_index()

    out = (other
           .merge(primary, on="agent", how="outer")
           .merge(coverage, on="agent", how="outer")
           .set_index("agent"))
    out["n_tasks_total"] = n_tasks_total
    return out


def normalize(matrix):
    """Min-max each dimension into [0, 1] with higher = better.
    NaN cells are filled with 0 (worst) so failed agents are penalized but
    still ranked. The Pareto-optimal extraction in `pareto_optimal()` and
    the per-row exclusion flag in `recommend()` surface the NaN-driven cells
    separately for transparent reporting.
    """
    out = matrix.copy()
    for _, col, direction in DIMENSIONS:
        v = out[col].astype(float)
        lo, hi = v.min(skipna=True), v.max(skipna=True)
        if pd.isna(lo) or pd.isna(hi) or hi == lo:
            out[col] = 0.5
            continue
        norm = (v - lo) / (hi - lo)
        if direction < 0:
            norm = 1.0 - norm
        out[col] = norm.fillna(0.0)
    return out


# ============================================================================
# MCDM methods
# ============================================================================

def wsm(normalized, weights):
    score = pd.Series(0.0, index=normalized.index)
    for name, col, _ in DIMENSIONS:
        score += weights[name] * normalized[col]
    return score.sort_values(ascending=False)


def topsis(normalized, weights):
    cols = [c for _, c, _ in DIMENSIONS]
    weighted = normalized[cols].copy()
    for name, col, _ in DIMENSIONS:
        weighted[col] = weighted[col] * weights[name]
    M = weighted.values
    ideal      = M.max(axis=0)
    anti_ideal = M.min(axis=0)
    d_plus  = np.sqrt(((M - ideal)      ** 2).sum(axis=1))
    d_minus = np.sqrt(((M - anti_ideal) ** 2).sum(axis=1))
    score = d_minus / (d_plus + d_minus + 1e-12)
    return pd.Series(score, index=normalized.index).sort_values(ascending=False)


def promethee(normalized, weights):
    """PROMETHEE-II net flow with linear preference function."""
    cols = [c for _, c, _ in DIMENSIONS]
    M = normalized[cols].values
    n = M.shape[0]
    weight_vec = np.array([weights[name] for name, _, _ in DIMENSIONS])

    pos_flow = np.zeros(n)
    neg_flow = np.zeros(n)
    for i in range(n):
        diffs = M[i] - M  # (n, d) per-criterion preference of i over each other agent
        prefs = np.maximum(0.0, diffs)  # only positive preferences count
        agg   = (prefs * weight_vec).sum(axis=1)  # (n,)
        pos_flow[i] = agg.sum() / max(n - 1, 1)
        neg_flow[i] = (np.maximum(0.0, -diffs) * weight_vec).sum() / max(n - 1, 1)

    net = pos_flow - neg_flow
    return pd.Series(net, index=normalized.index).sort_values(ascending=False)


# ============================================================================
# Pareto frontier
# ============================================================================

def pareto_optimal(matrix):
    """Return rows that are not Pareto-dominated on the routing dimensions.
    NaN is treated as the worst value (so failed agents are dominated)."""
    out = matrix.copy()
    for _, col, direction in DIMENSIONS:
        if direction < 0:
            out[col] = -out[col].fillna(np.inf)  # higher is better, NaN → −∞
        else:
            out[col] = out[col].fillna(-np.inf)

    cols = [c for _, c, _ in DIMENSIONS]
    M = out[cols].values
    n = M.shape[0]
    is_opt = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_opt[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if (M[j] >= M[i]).all() and (M[j] > M[i]).any():
                is_opt[i] = False
                break
    return matrix.iloc[is_opt]


# ============================================================================
# Inter-method agreement
# ============================================================================

def kendall_tau_table(rankings_per_method):
    """Pairwise Kendall tau between method rankings.

    Each entry in `rankings_per_method` is a sorted Series (agents in order
    by score). We need each agent's position in each method to compare,
    not the trivially-sorted score ranks. Build a per-method position
    vector aligned to a common agent ordering before computing tau.
    """
    from scipy.stats import kendalltau
    methods = list(rankings_per_method.keys())
    common = sorted(set.intersection(*[set(r.index) for r in rankings_per_method.values()]))
    positions = {
        m: [list(rankings_per_method[m].index).index(agent) for agent in common]
        for m in methods
    }
    out = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for a in methods:
        for b in methods:
            tau, _ = kendalltau(positions[a], positions[b])
            out.loc[a, b] = round(tau, 3)
    return out


# ============================================================================
# Sensitivity sweep
# ============================================================================

def _uniform_complement_weights(target_dim, target_value):
    """Build a weight dict with `target_value` on `target_dim` and the
    remainder distributed uniformly over the other DIM_NAMES."""
    others = [n for n in DIM_NAMES if n != target_dim]
    rest = (1.0 - target_value) / len(others)
    w = {n: rest for n in others}
    w[target_dim] = target_value
    return w


def sensitivity_sweep(target_dim, summary_df=None, n_steps=21,
                      method="wsm", strict=False):
    """Vary the weight on `target_dim` from 0 to 1 in `n_steps` increments
    (default 0.05 step). At each value, distribute (1 - target_value)
    uniformly across the other six dimensions and record the resulting
    score per agent under `method`.

    Returns a dataframe with columns: target_weight, agent, score, rank.
    Sorted by (target_weight, rank).
    """
    if target_dim not in DIM_NAMES:
        raise ValueError(f"Unknown dimension: {target_dim}. Available: {DIM_NAMES}")
    if summary_df is None:
        summary_df = load_summary()

    method_funcs = {"wsm": wsm, "topsis": topsis, "promethee": promethee}
    if method not in method_funcs:
        raise ValueError(f"Unknown method: {method}")

    rows = []
    grid = np.linspace(0.0, 1.0, n_steps)
    for w_target in grid:
        weights = _uniform_complement_weights(target_dim, float(w_target))
        out = recommend(weights, summary_df, methods=(method,), strict=strict)
        ranking = out["rankings"][method]
        for rank, (agent, score) in enumerate(ranking.items(), 1):
            rows.append({
                "target_weight": round(float(w_target), 4),
                "agent": agent,
                "score": float(score),
                "rank": rank,
            })

    return pd.DataFrame(rows)


def find_top1_breakpoints(sweep_df):
    """Walk the sweep in order of target_weight and emit the weight values
    where the top-1 agent changes.

    Returns a list of (target_weight, prev_agent, new_agent) tuples.
    """
    top1 = (sweep_df[sweep_df["rank"] == 1]
            .sort_values("target_weight")
            .reset_index(drop=True))
    breakpoints = []
    prev_agent = None
    for _, row in top1.iterrows():
        if prev_agent is not None and row["agent"] != prev_agent:
            breakpoints.append((row["target_weight"], prev_agent, row["agent"]))
        prev_agent = row["agent"]
    return breakpoints


def plot_sensitivity(sweep_df, target_dim, output_path, method="wsm"):
    """Save a PNG line plot of agent scores across the swept target weight."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = sweep_df.pivot(index="target_weight", columns="agent", values="score")
    fig, ax = plt.subplots(figsize=(10, 6))
    for agent in pivot.columns:
        ax.plot(pivot.index, pivot[agent], marker="o", markersize=3, label=agent)
    ax.set_xlabel(f"Weight on {target_dim} (others uniform)")
    ax.set_ylabel(f"{method.upper()} score")
    ax.set_title(f"Sensitivity to {target_dim} weight ({method.upper()})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ============================================================================
# CLI
# ============================================================================

def parse_weights(s):
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        out[k.strip()] = float(v)
    total = sum(out.values())
    if abs(total - 1.0) > 0.01:
        log(f"WARNING: weights sum to {total:.3f}, not 1.0; normalizing")
        out = {k: v / total for k, v in out.items()}
    for n in DIM_NAMES:
        if n not in out:
            raise ValueError(f"Missing weight for dimension: {n}")
    return out


def recommend(weights, summary_df=None, methods=("wsm", "topsis", "promethee"),
              strict=False):
    """Run MCDM recommendation across the requested methods.

    If `strict=True`, exclude any agent whose n_tasks_successful is less than
    n_tasks_total before ranking. This avoids the per-agent dimension means
    being computed over different sets of tasks (a bias that advantages agents
    with partial task success).
    """
    if summary_df is None:
        summary_df = load_summary()
    matrix = build_routing_matrix(summary_df)

    excluded = matrix[matrix["n_tasks_successful"] < matrix["n_tasks_total"]].index.tolist()
    if strict:
        active_matrix = matrix[matrix["n_tasks_successful"] == matrix["n_tasks_total"]]
    else:
        active_matrix = matrix

    normalized = normalize(active_matrix)
    method_funcs = {"wsm": wsm, "topsis": topsis, "promethee": promethee}
    results = {m: method_funcs[m](normalized, weights) for m in methods}
    pareto = pareto_optimal(active_matrix)

    return {
        "matrix":              matrix,
        "active_matrix":       active_matrix,
        "normalized":          normalized,
        "rankings":            results,
        "pareto":              pareto,
        "partial_coverage":    excluded,
        "strict":              strict,
    }


def _print_report(out, weights, methods):
    print("\n" + "=" * 72)
    print("PrismBench: preference-aware routing")
    print("=" * 72)
    print("\nWeights:")
    for n in DIM_NAMES:
        print(f"  {n:<12} = {weights[n]:.3f}")

    matrix = out["matrix"]
    print("\nCoverage (successful tasks per agent):")
    for agent, row in matrix.iterrows():
        succ = int(row["n_tasks_successful"]) if pd.notna(row["n_tasks_successful"]) else 0
        total = int(row["n_tasks_total"])
        flag = "" if succ == total else "  PARTIAL"
        print(f"  {agent:<18} {succ}/{total}{flag}")

    if out["strict"]:
        print(f"\nStrict mode: excluding {len(out['partial_coverage'])} partial-coverage agents from ranking")
        if out["partial_coverage"]:
            print("  Excluded: " + ", ".join(out["partial_coverage"]))

    print(f"\nPareto-optimal agents ({len(out['pareto'])}/{len(out['active_matrix'])}):")
    print("  " + ", ".join(out["pareto"].index.tolist()))

    for method, ranking in out["rankings"].items():
        print(f"\n--- {method.upper()} ranking ---")
        for rank, (agent, score) in enumerate(ranking.items(), 1):
            partial = "  (partial)" if agent in out["partial_coverage"] else ""
            print(f"  #{rank:<2} {agent:<18} score = {score:.4f}{partial}")

    if len(methods) > 1:
        print("\n--- Kendall tau between methods ---")
        tau = kendall_tau_table(out["rankings"])
        print(tau.to_string())
    print()


def main():
    parser = argparse.ArgumentParser(description="PrismBench MCDM router")
    parser.add_argument("--preset", choices=list(PRESETS.keys()),
                        help="Use a preset weight profile")
    parser.add_argument("--weights", type=str,
                        help="Custom weights, e.g. 'D1=0.4,D2=0.1,...'")
    parser.add_argument("--method", choices=["wsm", "topsis", "promethee"],
                        help="Run only one method (default: all)")
    parser.add_argument("--all-methods", action="store_true",
                        help="Run all three methods (default)")
    parser.add_argument("--csv", type=str,
                        help="Override path to master_scorecard_summary.csv")
    parser.add_argument("--strict", action="store_true",
                        help="Exclude agents with partial task coverage from ranking")
    parser.add_argument("--sweep", type=str, default=None,
                        choices=DIM_NAMES + ["all"],
                        help="Sensitivity sweep on the named dimension (or 'all' for every dimension)")
    parser.add_argument("--sweep-steps", type=int, default=21,
                        help="Number of weight values from 0 to 1 in the sweep (default 21 = 0.05 step)")
    parser.add_argument("--plot", action="store_true",
                        help="Save a PNG plot of the sweep to results/sensitivity_<dim>_<method>.png")
    args = parser.parse_args()

    if args.weights:
        weights = parse_weights(args.weights)
    elif args.preset:
        weights = PRESETS[args.preset]
    else:
        weights = PRESETS["balanced"]

    methods = ("wsm", "topsis", "promethee") if not args.method else (args.method,)
    summary = load_summary(args.csv) if args.csv else load_summary()

    if args.sweep is not None:
        method = args.method or "wsm"
        targets = DIM_NAMES if args.sweep == "all" else [args.sweep]
        for target in targets:
            print("\n" + "=" * 72)
            print(f"Sensitivity sweep on {target} (method = {method.upper()}, "
                  f"{args.sweep_steps} steps, strict = {args.strict})")
            print("=" * 72)
            sweep_df = sensitivity_sweep(target, summary,
                                         n_steps=args.sweep_steps,
                                         method=method, strict=args.strict)
            top1 = sweep_df[sweep_df["rank"] == 1][["target_weight", "agent", "score"]]
            print(top1.to_string(index=False))
            bps = find_top1_breakpoints(sweep_df)
            if bps:
                print("\nTop-1 breakpoints (weight value where the winner changes):")
                for w, prev, new in bps:
                    print(f"  at {target} = {w:.3f}: {prev}  ->  {new}")
            else:
                print("\nNo top-1 changes across the sweep (one agent dominates throughout).")

            if args.plot:
                FIGURES_DIR.mkdir(parents=True, exist_ok=True)
                out_path = FIGURES_DIR / f"sensitivity_{target}_{method}.png"
                plot_sensitivity(sweep_df, target, out_path, method=method)
                print(f"\nPlot saved: {out_path}")
        return

    out = recommend(weights, summary, methods=methods, strict=args.strict)
    _print_report(out, weights, methods)


if __name__ == "__main__":
    main()
