"""
Statistical significance testing for the scorecard.

Implements:
- Friedman test (non-parametric ANOVA across agents on repeated blocks)
- Nemenyi post-hoc pairwise comparisons
- Critical Difference value (Nemenyi cutoff)
- Text Critical Difference (CD) diagram

Operates on master_scorecard.csv. By default tests the strict-coverage
agent set (6 agents that succeeded on all 3 task types) over the 9
(task, run) blocks, using the per-task min-max normalized primary metric
so different metric types (F1 weighted, RMSE) are commensurate.

Usage:
    python -m src.stats                          # global Friedman + Nemenyi on D1
    python -m src.stats --metric D5_cost_usd     # significance on cost
    python -m src.stats --per-task               # also run per-task Friedman tests
"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import RESULTS_DIR


# Two agent sets used in the tests:
#
# STRICT_AGENTS: succeeded on at least one run of every task type. Useful for
# the routing comparison but Friedman drops blocks where any of these have
# NaN, so block count shrinks from 9 to 5.
#
# FULLY_COMPLETE_AGENTS: produced a valid primary metric on every (task, run)
# cell. Friedman can use all 9 blocks. Smaller agent set but more power per
# block.
STRICT_AGENTS = ["autogen", "autogluon", "claude_api_raw",
                 "claude_code", "langgraph", "pycaret"]
FULLY_COMPLETE_AGENTS = ["autogluon", "claude_api_raw", "claude_code", "pycaret"]
MAIN_TASKS = ["HD-PRED-01", "TAXI-PRED-01", "AR-PRED-01"]

# Studentized range distribution at alpha = 0.05 for Nemenyi critical difference.
# Source: Demsar (2006) "Statistical Comparisons of Classifiers over Multiple
# Data Sets", Table 5. Indexed by k (number of treatments).
NEMENYI_Q_ALPHA_05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
    7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}


# ============================================================================
# Data preparation
# ============================================================================

def load_detailed(path=None):
    p = Path(path) if path else (RESULTS_DIR / "master_scorecard.csv")
    if not p.exists():
        raise FileNotFoundError(f"{p} not found, run `python -m src.scorecard` first")
    return pd.read_csv(p)


def primary_metric_normalized(df, tasks=None, agents=None):
    """Return a wide-format frame with rows = (task, run) and columns = agents,
    values = per-task min-max normalized primary metric (higher = better).

    Drops blocks (rows) where any selected agent is missing the metric, so
    Friedman can run on complete data.
    """
    if tasks is None:
        tasks = MAIN_TASKS
    if agents is None:
        agents = STRICT_AGENTS

    sub = df[df.task_id.isin(tasks) & df.agent.isin(agents)].copy()

    sub["primary_raw"] = sub["D1_f1"]
    is_regression = sub["primary_raw"].isna() & sub["D1_rmse"].notna()
    sub.loc[is_regression, "primary_raw"] = 1.0 / (1.0 + sub.loc[is_regression, "D1_rmse"])

    norm_pieces = []
    for task in tasks:
        chunk = sub[sub.task_id == task].copy()
        v = chunk["primary_raw"]
        lo, hi = v.min(skipna=True), v.max(skipna=True)
        if pd.notna(lo) and pd.notna(hi) and hi > lo:
            chunk["primary_norm"] = (v - lo) / (hi - lo)
        else:
            chunk["primary_norm"] = np.nan
        norm_pieces.append(chunk)

    norm = pd.concat(norm_pieces)
    wide = norm.pivot_table(index=["task_id", "run_id"], columns="agent",
                            values="primary_norm")
    return wide.dropna(how="any")


def metric_wide(df, metric, tasks=None, agents=None, lower_is_better=False):
    """Generic wide-format pivot for any metric. Used for D5_cost_usd etc.
    If lower_is_better, values are flipped so Friedman ranks consistently
    treat higher = better (the test is rank-based so direction matters).
    """
    if tasks is None:
        tasks = MAIN_TASKS
    if agents is None:
        agents = STRICT_AGENTS

    sub = df[df.task_id.isin(tasks) & df.agent.isin(agents)].copy()
    wide = sub.pivot_table(index=["task_id", "run_id"], columns="agent", values=metric)
    wide = wide.dropna(how="any")
    if lower_is_better:
        wide = -wide  # flip sign so higher rank = better
    return wide


# ============================================================================
# Friedman test
# ============================================================================

def friedman(wide):
    """Run Friedman chi-square test on a wide-format frame (rows = blocks,
    columns = treatments).

    Returns (chi2_statistic, p_value, n_blocks, k_treatments).
    """
    from scipy.stats import friedmanchisquare
    columns = [wide[col].values for col in wide.columns]
    chi2, p = friedmanchisquare(*columns)
    return chi2, p, len(wide), len(wide.columns)


# ============================================================================
# Nemenyi post-hoc + Critical Difference
# ============================================================================

def nemenyi(wide, alpha=0.05):
    """Compute mean ranks per agent and the Nemenyi Critical Difference.

    Returns (rank_means_series, critical_difference, q_alpha).
    Two agents differ significantly at level `alpha` iff |rank_a - rank_b| > CD.
    """
    if alpha != 0.05:
        raise NotImplementedError("Only alpha = 0.05 is provisioned (extend NEMENYI_Q_ALPHA_05 for others)")
    ranks = wide.rank(axis=1, ascending=False, method="average")  # rank 1 = best
    rank_means = ranks.mean(axis=0).sort_values()
    n, k = len(wide), len(wide.columns)
    q = NEMENYI_Q_ALPHA_05.get(k)
    if q is None:
        raise ValueError(f"No q_alpha entry for k = {k}; extend NEMENYI_Q_ALPHA_05")
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    return rank_means, cd, q


def pairwise_nemenyi_significance(rank_means, cd):
    """Pairwise non-significance matrix: True iff |rank_a - rank_b| <= CD,
    meaning the two agents are NOT significantly different at alpha = 0.05.
    """
    agents = list(rank_means.index)
    matrix = pd.DataFrame(index=agents, columns=agents, dtype=bool)
    for a in agents:
        for b in agents:
            matrix.loc[a, b] = abs(rank_means[a] - rank_means[b]) <= cd
    return matrix


def wilcoxon_pairwise(wide, alpha=0.05, correction="bonferroni"):
    """Pairwise Wilcoxon signed-rank test on the paired differences between
    each pair of agents. More sensitive than Nemenyi to magnitude differences
    when ranks are consistently ordered across blocks.

    Returns a (raw_p, adjusted_p, win_count) tuple of DataFrames.
    `win_count` is the number of blocks where the row agent beat the column
    agent (i.e. higher value, since `wide` is already direction-corrected).

    Bonferroni adjusts each p by k*(k-1)/2 to control family-wise error rate.
    Use `correction=None` to skip.
    """
    from scipy.stats import wilcoxon
    agents = list(wide.columns)
    k = len(agents)
    n_pairs = k * (k - 1) // 2

    raw = pd.DataFrame(index=agents, columns=agents, dtype=float)
    win = pd.DataFrame(index=agents, columns=agents, dtype=int)
    for a in agents:
        for b in agents:
            if a == b:
                raw.loc[a, b] = 1.0
                win.loc[a, b] = 0
                continue
            diffs = wide[a] - wide[b]
            wins_ab = int((diffs > 0).sum())
            win.loc[a, b] = wins_ab
            nonzero = diffs[diffs != 0]
            if len(nonzero) < 2:
                raw.loc[a, b] = 1.0
                continue
            try:
                _, p = wilcoxon(nonzero)
                raw.loc[a, b] = p
            except ValueError:
                raw.loc[a, b] = 1.0

    if correction == "bonferroni":
        adj = (raw * n_pairs).clip(upper=1.0)
    else:
        adj = raw.copy()

    return raw, adj, win


def text_wilcoxon_report(wide, alpha=0.05):
    """Print a Wilcoxon pairwise report. Each cell shows the row agent's
    win count over the column agent and the Bonferroni-adjusted p value.
    """
    raw, adj, win = wilcoxon_pairwise(wide, alpha=alpha, correction="bonferroni")
    agents = list(wide.columns)
    n = len(wide)
    print(f"\nWilcoxon signed-rank pairwise (Bonferroni adjusted p, n = {n} blocks per pair):")
    print(f"  Cell shows 'wins/n  p_adj'; bold if p_adj < {alpha}.")
    header = "                  " + "  ".join(f"{a[:12]:>12}" for a in agents)
    print(header)
    for a in agents:
        cells = []
        for b in agents:
            if a == b:
                cells.append(f"{'-':>12}")
                continue
            w = int(win.loc[a, b])
            p = float(adj.loc[a, b])
            mark = "*" if p < alpha else " "
            cells.append(f"{w:>2}/{n}  {p:.4f}{mark}".rjust(12))
        print(f"  {a:<14}  " + "  ".join(cells))
    print(f"\n  '*' marks Bonferroni-adjusted p < {alpha} (significant)")


# ============================================================================
# Reporting
# ============================================================================

def text_cd_diagram(rank_means, cd, label="primary metric"):
    """Print a text Critical Difference summary."""
    sorted_means = rank_means.sort_values()
    print(f"\nCritical Difference (alpha = 0.05): {cd:.3f}")
    print(f"Mean ranks on {label} (lower rank = better; n = ranks averaged across blocks):")
    for agent, r in sorted_means.items():
        print(f"  {agent:<18}  rank = {r:.3f}")

    print("\nPairwise non-significance (agents within CD of each other; ns = not sig):")
    sig_matrix = pairwise_nemenyi_significance(rank_means, cd)
    agents = list(sorted_means.index)
    print("                  " + "  ".join(f"{a[:10]:>10}" for a in agents))
    for a in agents:
        cells = []
        for b in agents:
            cells.append("ns" if sig_matrix.loc[a, b] else "sig")
        print(f"  {a:<14}  " + "  ".join(f"{c:>10}" for c in cells))

    print("\nGroups of agents NOT pairwise-significantly-different at alpha = 0.05:")
    groups = []
    visited = set()
    for a in agents:
        if a in visited: continue
        group = {a}
        for b in agents:
            if a == b: continue
            if sig_matrix.loc[a, b]:
                group.add(b)
        groups.append(sorted(group))
        visited.update(group)
    for g in groups:
        marker = " (singleton, sig diff from all)" if len(g) == 1 else ""
        print(f"  {{{', '.join(g)}}}{marker}")


def report(metric_label, wide, lower_is_better=False, wilcoxon=True):
    print("\n" + "=" * 72)
    print(f"Friedman + Nemenyi on {metric_label}")
    print("=" * 72)
    print(f"Blocks (task x run) considered: {len(wide)}")
    print(f"Agents considered: {list(wide.columns)}")
    chi2, p, n, k = friedman(wide)
    print(f"\nFriedman chi2 = {chi2:.3f}, p = {p:.4g}, n = {n} blocks, k = {k} agents")
    if p >= 0.05:
        print("  Friedman p >= 0.05: cannot reject null that agent ranks are equal.")
        print("  Skipping Nemenyi (not warranted).")
        return
    print("  Friedman p < 0.05: at least one agent differs significantly. Running Nemenyi.")
    rank_means, cd, _ = nemenyi(wide)
    label_dir = "(higher metric = better)" if not lower_is_better else "(lower metric = better, signs flipped for ranking)"
    text_cd_diagram(rank_means, cd, label=f"{metric_label} {label_dir}")

    if wilcoxon:
        print("\n" + "-" * 72)
        print("Wilcoxon signed-rank pairwise (more sensitive than Nemenyi to magnitudes)")
        print("-" * 72)
        text_wilcoxon_report(wide)


def main():
    parser = argparse.ArgumentParser(description="Statistical significance tests")
    parser.add_argument("--metric", default="D1",
                        help="Metric to test. 'D1' uses normalized primary; otherwise pass a column from master_scorecard.csv (e.g. D5_cost_usd, D2_pylint).")
    parser.add_argument("--agents", default="strict",
                        choices=["strict", "complete"],
                        help="'strict' = 6 agents that succeeded on every task type at least once (5 blocks survive). 'complete' = 4 agents with full 9-block coverage (more power).")
    parser.add_argument("--per-task", action="store_true",
                        help="Also run a Friedman test per task on raw primary metric")
    parser.add_argument("--no-wilcoxon", action="store_true",
                        help="Skip the Wilcoxon signed-rank pairwise report")
    parser.add_argument("--csv", type=str, help="Override path to master_scorecard.csv")
    args = parser.parse_args()

    df = load_detailed(args.csv) if args.csv else load_detailed()
    agent_set = STRICT_AGENTS if args.agents == "strict" else FULLY_COMPLETE_AGENTS
    print(f"\nAgent set: {args.agents} ({len(agent_set)} agents): {agent_set}")

    do_wilcoxon = not args.no_wilcoxon
    if args.metric == "D1":
        wide = primary_metric_normalized(df, agents=agent_set)
        report("D1 normalized primary metric", wide, lower_is_better=False, wilcoxon=do_wilcoxon)
    else:
        lower_better_cols = {"D5_cost_usd", "D5_carbon_kg", "D4_time_sec", "D5_tokens", "D6_cv"}
        lib = args.metric in lower_better_cols
        wide = metric_wide(df, args.metric, agents=agent_set, lower_is_better=lib)
        report(args.metric, wide, lower_is_better=lib, wilcoxon=do_wilcoxon)

    if args.per_task:
        print("\n" + "=" * 72)
        print("Per-task Friedman tests on D1 primary metric (n = 3 runs each)")
        print("=" * 72)
        for task in MAIN_TASKS:
            sub_wide = metric_wide(df, "D1_f1" if task != "TAXI-PRED-01" else "D1_rmse",
                                   tasks=[task],
                                   lower_is_better=(task == "TAXI-PRED-01"))
            if len(sub_wide) < 2:
                print(f"\n{task}: insufficient blocks ({len(sub_wide)}); skipping")
                continue
            chi2, p, n, k = friedman(sub_wide)
            print(f"\n{task}: chi2 = {chi2:.3f}, p = {p:.4g}, n = {n}, k = {k}")
            if p < 0.05 and k <= 9:
                rank_means, cd, _ = nemenyi(sub_wide)
                print(f"  CD = {cd:.3f}")
                for a, r in rank_means.items():
                    print(f"    {a:<16} rank = {r:.3f}")


if __name__ == "__main__":
    main()
