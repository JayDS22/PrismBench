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
    scorecards = []
    for scorecard_path in RESULTS_DIR.rglob("scorecard.json"):
        try:
            scorecards.append(load_json(scorecard_path))
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
        "D3_auto_score": d3.get("auto_score"),
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