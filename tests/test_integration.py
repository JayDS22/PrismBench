"""Integration tests for the PrismBench end-to-end pipeline.

These tests do not run live agents (no API keys, no network). Instead they
exercise the full chain from `master_scorecard_summary.csv` through the router
and assert the headline claims of the report still hold against the committed
results.
"""
from pathlib import Path

import pandas as pd
import pytest

from src import router
import prismbench_utils as pbu


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Cache fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def summary_df():
    """The per-(agent, task) summary the router consumes."""
    path = RESULTS / "master_scorecard_summary.csv"
    if not path.exists():
        pytest.skip(f"{path} missing; run `python -m src.scorecard` first")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def detailed_df():
    """One row per run, for the headline-claim spot-checks."""
    path = RESULTS / "master_scorecard.csv"
    if not path.exists():
        pytest.skip(f"{path} missing; run `python -m src.scorecard` first")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def test_pipeline_produces_full_recommendation(summary_df):
    """summary CSV -> routing matrix -> normalize -> recommend.
    Every preset should yield a top agent for every method."""
    matrix = router.build_routing_matrix(summary_df)
    assert matrix.shape[0] == 9, f"expected 9 agents, got {matrix.shape[0]}"

    for preset_name, weights in router.PRESETS.items():
        rec = router.recommend(weights)
        for method in ("wsm", "topsis", "promethee"):
            ranking = rec["rankings"][method]
            assert len(ranking) == 9
            assert not ranking.isna().any(), (
                f"NaN scores in {preset_name}/{method}"
            )
            top_agent = str(ranking.index[0])
            assert top_agent in matrix.index, (
                f"top agent {top_agent!r} not in matrix"
            )


def test_balanced_preset_top1_is_claude_code_or_autogluon(summary_df):
    """The balanced preset should pick claude_code (Section 4.3 headline)
    under all three methods. autogluon is acceptable if scoring shifts
    enough to flip; either way the top must come from this small set."""
    rec = router.recommend(router.PRESETS["balanced"])
    for method in ("wsm", "topsis", "promethee"):
        top = str(rec["rankings"][method].index[0])
        assert top in {"claude_code", "autogluon"}, (
            f"balanced/{method} top1 = {top!r}, expected claude_code or autogluon"
        )


def test_proposition_1_holds_on_real_data(summary_df):
    """Theoretical Proposition 1 says WSM and PROMETHEE-II rank identically.
    Verify Kendall tau == 1.0 across all six shipped presets, on the actual
    benchmark scorecard."""
    matrix = router.build_routing_matrix(summary_df)
    normalized = router.normalize(matrix)
    for preset_name, weights in router.PRESETS.items():
        wsm_rank = router.wsm(normalized, weights).rank(ascending=False)
        prom_rank = router.promethee(normalized, weights).rank(ascending=False)
        tau = router.kendall_tau_table(
            {"wsm": wsm_rank, "promethee": prom_rank}
        ).loc["wsm", "promethee"]
        assert tau == pytest.approx(1.0), (
            f"preset {preset_name!r}: WSM/PROMETHEE tau = {tau}, expected 1.0"
        )


# ---------------------------------------------------------------------------
# Headline empirical claims (Sections 4.1, 4.2)
# ---------------------------------------------------------------------------

def test_headline_f1_claude_code_hd_pred(detailed_df):
    """Section 4.1: claude_code mean F1 on HD-PRED-01 is 0.9017 +- 0.001."""
    sub = detailed_df[
        (detailed_df.agent == "claude_code")
        & (detailed_df.task_id == "HD-PRED-01")
    ]
    assert len(sub) == 3
    assert sub.D1_f1.mean() == pytest.approx(0.9017, abs=0.001)


def test_headline_autogluon_taxi_rmse(detailed_df):
    """Section 4.1: autogluon mean RMSE on TAXI-PRED-01 is 0.667 +- 0.005."""
    sub = detailed_df[
        (detailed_df.agent == "autogluon")
        & (detailed_df.task_id == "TAXI-PRED-01")
    ]
    assert len(sub) == 3
    assert sub.D1_rmse.mean() == pytest.approx(0.667, abs=0.005)


def test_rq4_cost_ratio_at_least_10x(detailed_df):
    """RQ4 headline (Section 4.2): claude_api_raw is at least 10x more
    expensive than claude_code on the main benchmark, averaged over all
    9 main cells."""
    main_tasks = ["HD-PRED-01", "TAXI-PRED-01", "AR-PRED-01"]
    cc_cost = detailed_df[
        (detailed_df.agent == "claude_code")
        & (detailed_df.task_id.isin(main_tasks))
    ].D5_cost_usd.mean()
    cr_cost = detailed_df[
        (detailed_df.agent == "claude_api_raw")
        & (detailed_df.task_id.isin(main_tasks))
    ].D5_cost_usd.mean()

    assert cc_cost > 0
    ratio = cr_cost / cc_cost
    assert ratio >= 10.0, (
        f"cost ratio {ratio:.2f}x falls below the 10x claim in the report"
    )


# ---------------------------------------------------------------------------
# Utils facade smoke test
# ---------------------------------------------------------------------------

def test_prismbench_utils_top_agent_matches_router():
    """The convenience facade and the underlying router agree."""
    direct = router.recommend(router.PRESETS["balanced"])
    direct_top = str(direct["rankings"]["wsm"].index[0])
    facade_top = pbu.top_agent(pbu.PRESETS["balanced"], method="wsm")
    assert direct_top == facade_top
