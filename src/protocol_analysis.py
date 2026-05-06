"""
Cross-protocol stability analysis for the multi-protocol study.

Operates on master_scorecard.csv after Protocol A (same-prompt rule),
Protocol B (paradigm-native, scoped subset), and Protocol C (strict-
explicit) runs are all complete.

Implements:
- Procrustes analysis between per-agent positions across protocols
- Two-way ANOVA decomposing variance between agent identity and protocol
- Cross-protocol Kendall tau on agent rankings

Usage:
    python -m src.protocol_analysis
"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import RESULTS_DIR, log


# Mapping from Protocol C task IDs to their Protocol A counterparts so
# we can compare like-with-like.
PROTOCOL_A_TASKS = ["HD-PRED-01", "TAXI-PRED-01", "AR-PRED-01"]
PROTOCOL_C_TASKS = ["HD-PRED-STRICT-01", "TAXI-PRED-STRICT-01", "AR-PRED-STRICT-01"]
TASK_PAIRS = list(zip(PROTOCOL_A_TASKS, PROTOCOL_C_TASKS))

# Routing dimensions used for Procrustes (the 7 we used in the router).
# Column names are as they appear in the per-agent matrix produced by
# _per_agent_matrix() (primary is the per-task normalized D1 metric).
DIM_COLS = [
    ("primary",       False, "primary"),
    ("D2_combined",   False, "D2"),
    ("D3_combined",   False, "D3"),
    ("D4_time_sec",   True,  "D4_speed"),
    ("D5_cost_usd",   True,  "D5_cost"),
    ("D5_carbon_kg",  True,  "D5_carbon"),
    ("D6_cv",         True,  "D6_robust"),
]


def _per_agent_matrix(df, tasks):
    """Build a per-agent matrix (rows = agent, cols = routing dims) for a
    given set of tasks. Same logic as router.build_routing_matrix but
    parameterized by task set.
    """
    sub = df[df.task_id.isin(tasks)].copy()
    sub["primary_raw"] = sub["D1_f1"]
    is_reg = sub["primary_raw"].isna() & sub["D1_rmse"].notna()
    sub.loc[is_reg, "primary_raw"] = 1.0 / (1.0 + sub.loc[is_reg, "D1_rmse"])

    primary_pieces = []
    for task in tasks:
        chunk = sub[sub.task_id == task].copy()
        v = chunk["primary_raw"]
        lo, hi = v.min(skipna=True), v.max(skipna=True)
        if pd.notna(lo) and pd.notna(hi) and hi > lo:
            chunk["primary_norm"] = (v - lo) / (hi - lo)
        else:
            chunk["primary_norm"] = np.nan
        primary_pieces.append(chunk[["agent", "primary_norm"]])
    primary = pd.concat(primary_pieces).groupby("agent")["primary_norm"].mean()

    other_cols = ["D2_combined", "D3_combined", "D4_time_sec",
                  "D5_cost_usd", "D5_carbon_kg", "D6_cv"]
    other = sub.groupby("agent")[other_cols].mean()
    out = other.copy()
    out["primary"] = primary
    return out


def _normalize_columns(df, cols_with_dir):
    """Min-max normalize each named column to [0, 1] with higher = better.
    NaN cells are filled with 0."""
    out = df.copy()
    for col, lower_better, _label in cols_with_dir:
        v = out[col].astype(float)
        lo, hi = v.min(skipna=True), v.max(skipna=True)
        if pd.isna(lo) or pd.isna(hi) or hi == lo:
            out[col] = 0.5
            continue
        norm = (v - lo) / (hi - lo)
        if lower_better:
            norm = 1.0 - norm
        out[col] = norm.fillna(0.0)
    return out


def procrustes_protocols(matrix_a, matrix_c, dim_cols=DIM_COLS):
    """Procrustes-style alignment between protocol-A and protocol-C
    per-agent positions. Computes the standardized disparity (sum of
    squared distances after optimal rotation+scale+translation).

    Lower disparity = the protocols produce more similar agent geometries.
    """
    from scipy.spatial import procrustes
    common = sorted(set(matrix_a.index) & set(matrix_c.index))
    cols = [c for c, _, _ in dim_cols]
    # rows = agents, cols = routing dimensions
    a = _normalize_columns(matrix_a.loc[common], dim_cols)[cols].values
    c = _normalize_columns(matrix_c.loc[common], dim_cols)[cols].values
    _, _, disparity = procrustes(a, c)
    return float(disparity), common


def kendall_tau_per_protocol(detailed_df, protocol_tasks_dict, metric="D1_f1"):
    """Compute Kendall tau between agent rankings under each protocol.

    detailed_df: master_scorecard.csv rows.
    protocol_tasks_dict: {"A": [task_ids], "C": [task_ids], ...}.
    """
    from scipy.stats import kendalltau

    rankings = {}
    for proto, tasks in protocol_tasks_dict.items():
        sub = detailed_df[detailed_df.task_id.isin(tasks)]
        # Score per agent: mean primary across tasks, normalized
        m = _per_agent_matrix(sub, tasks)
        rankings[proto] = m["primary"].sort_values(ascending=False)

    common = set.intersection(*(set(r.index) for r in rankings.values()))
    common = sorted(common)
    out = {}
    for p1 in rankings:
        for p2 in rankings:
            if p1 >= p2:
                continue
            ranks1 = [list(rankings[p1].index).index(a) for a in common]
            ranks2 = [list(rankings[p2].index).index(a) for a in common]
            tau, p = kendalltau(ranks1, ranks2)
            out[f"{p1}_vs_{p2}"] = (round(tau, 3), round(p, 4))
    return out, rankings


def two_way_anova(detailed_df, tasks_a, tasks_c, metric="D1_f1"):
    """Two-way ANOVA on the primary metric with factors (agent, protocol).
    Returns dict of F-statistics and p-values for each main effect plus
    interaction. Uses statsmodels if available, scipy fallback otherwise.
    """
    rows = []
    for proto, tasks in [("A", tasks_a), ("C", tasks_c)]:
        sub = detailed_df[detailed_df.task_id.isin(tasks)].copy()
        # Use F1 if present, else 1/(1+RMSE)
        sub["primary"] = sub["D1_f1"]
        is_reg = sub["primary"].isna() & sub["D1_rmse"].notna()
        sub.loc[is_reg, "primary"] = 1.0 / (1.0 + sub.loc[is_reg, "D1_rmse"])
        # Per-task min-max normalize so different metrics are commensurate
        for task in sub["task_id"].unique():
            mask = sub["task_id"] == task
            v = sub.loc[mask, "primary"]
            lo, hi = v.min(skipna=True), v.max(skipna=True)
            if pd.notna(lo) and pd.notna(hi) and hi > lo:
                sub.loc[mask, "primary"] = (v - lo) / (hi - lo)
        sub = sub.dropna(subset=["primary"])
        sub["protocol"] = proto
        rows.append(sub[["agent", "protocol", "task_id", "run_id", "primary"]])
    long = pd.concat(rows, ignore_index=True)

    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        long["agent"] = long["agent"].astype("category")
        long["protocol"] = long["protocol"].astype("category")
        # Two-way: primary ~ agent + protocol + agent:protocol
        model = smf.ols("primary ~ C(agent) + C(protocol) + C(agent):C(protocol)",
                        data=long).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        return {
            "agent_F":      round(float(anova.loc["C(agent)",                 "F"]), 3),
            "agent_p":      round(float(anova.loc["C(agent)",                 "PR(>F)"]), 5),
            "protocol_F":   round(float(anova.loc["C(protocol)",              "F"]), 3),
            "protocol_p":   round(float(anova.loc["C(protocol)",              "PR(>F)"]), 5),
            "interaction_F": round(float(anova.loc["C(agent):C(protocol)",    "F"]), 3),
            "interaction_p": round(float(anova.loc["C(agent):C(protocol)",    "PR(>F)"]), 5),
            "n_obs":        int(len(long)),
        }, long
    except ImportError:
        # statsmodels not installed; fall back to a one-factor scipy ANOVA
        # on protocol main effect (loses interaction term).
        from scipy.stats import f_oneway
        groups = [long[long.protocol == p]["primary"].values for p in long.protocol.unique()]
        F, p = f_oneway(*groups)
        return {
            "protocol_F": round(float(F), 3),
            "protocol_p": round(float(p), 5),
            "agent_F":    None,
            "agent_p":    None,
            "interaction_F": None,
            "interaction_p": None,
            "n_obs":      int(len(long)),
            "note":       "scipy fallback (one-way ANOVA on protocol only); install statsmodels for two-way",
        }, long


def per_agent_protocol_gap(matrix_a, matrix_c, dim_col="primary"):
    """Per-agent absolute difference in the primary score across protocols.
    Positive value = better under Protocol C; negative = worse."""
    common = sorted(set(matrix_a.index) & set(matrix_c.index))
    out = pd.DataFrame({
        "protocol_A": matrix_a.loc[common, dim_col],
        "protocol_C": matrix_c.loc[common, dim_col],
    })
    out["gap_C_minus_A"] = (out["protocol_C"] - out["protocol_A"]).round(4)
    return out.sort_values("gap_C_minus_A")


def main():
    parser = argparse.ArgumentParser(description="Multi-protocol stability analysis")
    parser.add_argument("--csv", type=str, help="Override master_scorecard.csv path")
    args = parser.parse_args()

    csv = Path(args.csv) if args.csv else RESULTS_DIR / "master_scorecard.csv"
    if not csv.exists():
        print(f"ERROR: {csv} not found")
        sys.exit(1)
    df = pd.read_csv(csv)

    print("\n" + "=" * 72)
    print("Cross-protocol stability analysis (Protocols A vs C)")
    print("=" * 72)

    matrix_a = _per_agent_matrix(df, PROTOCOL_A_TASKS)
    matrix_c = _per_agent_matrix(df, PROTOCOL_C_TASKS)

    print(f"\nProtocol A agents: {sorted(matrix_a.index)}")
    print(f"Protocol C agents: {sorted(matrix_c.index)}")

    # Per-agent protocol gap on primary
    gap = per_agent_protocol_gap(matrix_a, matrix_c)
    print("\n--- Per-agent protocol gap (primary score, A vs C) ---")
    print(gap.to_string())

    # Procrustes
    disp, common = procrustes_protocols(matrix_a, matrix_c)
    print(f"\n--- Procrustes disparity (A vs C, {len(common)} agents) ---")
    print(f"  disparity = {disp:.4f}  (0 = identical geometries; 1 = maximally different)")
    print(f"  agents in common: {common}")

    # Kendall tau
    tau, rankings = kendall_tau_per_protocol(
        df, {"A": PROTOCOL_A_TASKS, "C": PROTOCOL_C_TASKS}
    )
    print(f"\n--- Cross-protocol Kendall tau on rankings ---")
    for k, (t, p) in tau.items():
        print(f"  {k}:  tau = {t:.3f}  (p = {p})")

    # ANOVA
    anova, long = two_way_anova(df, PROTOCOL_A_TASKS, PROTOCOL_C_TASKS)
    print(f"\n--- Two-way ANOVA on primary metric (factors: agent, protocol) ---")
    for k, v in anova.items():
        print(f"  {k}: {v}")
    print(f"  n_obs: {anova['n_obs']}")


if __name__ == "__main__":
    main()
