"""
Reads all results and builds the master comparison table.

Usage:
    python -m src.scorecard
    python -m src.scorecard --csv results/master_scorecard.csv
"""
import sys
import argparse
import pandas as pd
from pathlib import Path

from src.utils import RESULTS_DIR, load_json, log


def collect_scorecards():
    """Load every scorecard.json under results/. The path layout is
    `results/<agent>/<task_id>/run_<run_id>/scorecard.json`, so we override
    those three identity fields from the path — the JSON contents have
    historically held a bare "?" because wrappers couldn't see the canonical
    task_id passed by the runner.
    """
    scorecards = []
    for scorecard_path in RESULTS_DIR.rglob("scorecard.json"):
        try:
            sc = load_json(scorecard_path)
            parts = scorecard_path.relative_to(RESULTS_DIR).parts
            if len(parts) >= 3:
                sc["agent"]   = parts[0]
                sc["task_id"] = parts[1]
                run_dir = parts[2]
                if run_dir.startswith("run_"):
                    try:
                        sc["run_id"] = int(run_dir.split("_", 1)[1])
                    except ValueError:
                        pass
            scorecards.append(sc)
        except Exception as e:
            log(f"WARNING: failed to load {scorecard_path}: {e}")
    return scorecards


def flatten_scorecard(sc):
    """Flatten a nested scorecard dict into a single-level row for pandas."""
    d1 = sc.get("D1_accuracy",     {}) or {}
    d2 = sc.get("D2_code_quality", {}) or {}
    d3 = sc.get("D3_explainability", {}) or {}
    d4 = sc.get("D4_speed",        {}) or {}
    d5 = sc.get("D5_cost",         {}) or {}
    d6 = sc.get("D6_robustness",   {}) or {}

    # Combined D2/D3: average of the static-analysis score and the
    # LLM-judge score when both are present. The two often disagree (see
    # findings J1), so this is a routing convenience, not a truth signal.
    pl, jl = d2.get("pylint_score"), d2.get("llm_judge_score")
    d2_combined = (pl + jl) / 2 if (pl is not None and jl is not None) else None
    a3, j3 = d3.get("auto_score"), d3.get("llm_judge_score")
    d3_combined = (a3 + j3) / 2 if (a3 is not None and j3 is not None) else None

    return {
        "agent":      sc.get("agent"),
        "task_id":    sc.get("task_id"),
        "run_id":     sc.get("run_id"),
        "D1_f1":        d1.get("f1_weighted"),
        "D1_accuracy":  d1.get("accuracy"),
        "D1_auc_roc":   d1.get("auc_roc"),
        "D1_rmse":      d1.get("rmse"),
        "D1_r2":        d1.get("r2"),
        "D1_rubric":    d1.get("rubric_score"),
        "D1_type":      d1.get("type"),
        "D2_pylint":    d2.get("pylint_score"),
        "D2_llm_judge": d2.get("llm_judge_score"),
        "D2_combined":  round(d2_combined, 2) if d2_combined is not None else None,
        "D3_auto_score": d3.get("auto_score"),
        "D3_llm_judge": d3.get("llm_judge_score"),
        "D3_combined":  round(d3_combined, 2) if d3_combined is not None else None,
        "D3_shap":      d3.get("shap_generated"),
        "D4_time_sec":  d4.get("wall_clock_sec"),
        "D5_cost_usd":  d5.get("api_cost_usd"),
        "D5_tokens":    d5.get("tokens_total"),
        "D5_carbon_kg": d5.get("carbon_kg"),
        "D5_carbon_source": d5.get("carbon_source"),
        "D6_cv":        d6.get("cv"),
        "D6_std":       d6.get("std"),
    }


def build_scorecard(output_path=None):
    scorecards = collect_scorecards()
    if not scorecards:
        log("WARNING: no scorecards found in results/ — run some experiments first")
        return pd.DataFrame()

    df = pd.DataFrame([flatten_scorecard(sc) for sc in scorecards])
    df = df.sort_values(["task_id", "agent", "run_id"]).reset_index(drop=True)

    # Backfill D6 robustness (CV and std of D1) per (agent, task) from the
    # actual run data. The evaluator's score_robustness() needs all_run_scores
    # passed in, but task_runner.run_single() doesn't pass it — so D6_cv and
    # D6_std are NaN in scorecard.json files. Computing here from the
    # detailed dataframe is the cheapest fix and keeps the source of truth
    # in one place.
    primary = df["D1_f1"].fillna(df["D1_rmse"])  # F1 for classification, RMSE for regression
    df["_primary"] = primary
    grp = df.groupby(["agent", "task_id"])["_primary"]
    df["D6_std"] = grp.transform("std").round(4)
    mean = grp.transform("mean")
    df["D6_cv"] = (df["D6_std"] / mean.abs()).round(4)
    df = df.drop(columns=["_primary"])

    numeric_cols = [c for c in df.columns if c.startswith("D") and pd.api.types.is_numeric_dtype(df[c])]
    summary = df.groupby(["agent", "task_id"])[numeric_cols].mean().round(4).reset_index()

    log(
        f"\n{'='*70}\n"
        f"MASTER SCORECARD - {len(scorecards)} scorecards, {len(summary)} agent x task combos\n"
        f"{'='*70}"
    )
    print(summary.to_string(index=False))

    default_path   = Path(output_path) if output_path else RESULTS_DIR / "master_scorecard.csv"
    summary_path   = Path(str(default_path).replace(".csv", "_summary.csv"))

    df.to_csv(default_path, index=False)
    summary.to_csv(summary_path, index=False)
    log(f"Saved detailed : {default_path}")
    log(f"Saved summary  : {summary_path}")

    return summary


def print_agent_rankings(summary):
    if summary.empty:
        return

    # (column, ascending) - ascending=True means lower is better
    rankings = {
        "D1 Accuracy (F1)" : ("D1_f1",         False),
        "D2 Code Quality"  : ("D2_pylint",      False),
        "D3 Explainability": ("D3_auto_score",  False),
        "D4 Speed (time)"  : ("D4_time_sec",    True),
        "D5 Cost (USD)"    : ("D5_cost_usd",    True),
    }

    print(f"\n{'='*70}")
    print("AGENT RANKINGS BY DIMENSION")
    print(f"{'='*70}")

    for dim_name, (col, ascending) in rankings.items():
        if col not in summary.columns or not summary[col].notna().any():
            continue
        ranked = (
            summary.groupby("agent")[col]
            .mean()
            .sort_values(ascending=ascending)
            .reset_index()
        )
        print(f"\n  {dim_name}:")
        for i, row in ranked.iterrows():
            print(f"    {'#'+str(i+1):<4} {row['agent']:20s} = {row[col]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DS-RouteBench Scorecard")
    parser.add_argument("--csv", type=str, help="Output CSV path")
    args = parser.parse_args()
    summary = build_scorecard(args.csv)
    print_agent_rankings(summary)