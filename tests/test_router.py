"""Unit tests for the MCDM router math.

Goal: pin down the algebraic guarantees the report relies on (notably
Theoretical Proposition 1) so a refactor can't silently break them.
"""
import numpy as np
import pandas as pd
import pytest

from src import router


# ---------------------------------------------------------------------------
# Helper: build a small synthetic routing matrix with the same column names
# the production pipeline emits, so the tests exercise real schema.
# ---------------------------------------------------------------------------

def _toy_matrix() -> pd.DataFrame:
    """Three agents, seven dimensions, with hand-set values."""
    return pd.DataFrame(
        index=pd.Index(["alpha", "beta", "gamma"], name="agent"),
        data={
            "primary":      [0.90, 0.60, 0.30],   # higher better
            "D2_combined":  [8.0,  6.0,  4.0],    # higher better
            "D3_combined":  [9.0,  5.0,  1.0],    # higher better
            "D4_time_sec":  [10.0, 50.0, 100.0],  # lower better
            "D5_cost_usd":  [0.0,  0.05, 0.10],   # lower better
            "D5_carbon_kg": [0.0001, 0.001, 0.005],
            "D6_cv":        [0.02, 0.10, 0.20],   # lower better
            "n_tasks_successful": [3, 3, 3],
            "n_tasks_total":      [3, 3, 3],
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_presets_sum_to_one():
    """Every shipped preset weight vector must sum to 1.0 within float epsilon."""
    for name, weights in router.PRESETS.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9, f"preset {name!r} sums to {total}, not 1.0"


def test_normalize_minmax_basic():
    """min-max normalization maps to [0, 1] and inverts lower-is-better dims."""
    matrix = _toy_matrix()
    normalized = router.normalize(matrix)

    # Higher-is-better: alpha (best) -> 1, gamma (worst) -> 0.
    assert normalized.loc["alpha", "primary"] == pytest.approx(1.0)
    assert normalized.loc["gamma", "primary"] == pytest.approx(0.0)

    # Lower-is-better (cost): alpha (cheapest) -> 1, gamma (most expensive) -> 0.
    assert normalized.loc["alpha", "D5_cost_usd"] == pytest.approx(1.0)
    assert normalized.loc["gamma", "D5_cost_usd"] == pytest.approx(0.0)

    # Every value in the score columns is on [0, 1].
    score_cols = [c for c in normalized.columns
                  if c not in ("n_tasks_successful", "n_tasks_total")]
    for col in score_cols:
        assert (normalized[col] >= -1e-9).all(), f"{col} below 0"
        assert (normalized[col] <= 1.0 + 1e-9).all(), f"{col} above 1"


def test_wsm_unit_weight_recovers_dimension():
    """With weight 1 on D1 and 0 elsewhere, WSM score equals the
    normalized D1 column. Catches accidental column shuffling."""
    normalized = router.normalize(_toy_matrix())
    weights = {d: 0.0 for d in router.DIM_NAMES}
    weights["D1"] = 1.0

    score = router.wsm(normalized, weights)
    expected = normalized["primary"]

    for agent in score.index:
        assert score.loc[agent] == pytest.approx(expected.loc[agent])


def test_proposition_1_wsm_promethee_kendall_tau_one():
    """Theoretical Proposition 1: WSM and PROMETHEE-II with linear preference
    function produce identical rankings on min-max normalized scores. We assert
    Kendall tau == 1.0 across multiple weight regimes.
    """
    normalized = router.normalize(_toy_matrix())

    weight_vectors = [
        router.PRESETS["balanced"],
        router.PRESETS["accuracy"],
        router.PRESETS["frugal"],
        router.PRESETS["green"],
    ]
    for weights in weight_vectors:
        wsm_rank = router.wsm(normalized, weights).rank(ascending=False)
        prom_rank = router.promethee(normalized, weights).rank(ascending=False)

        rankings = {"wsm": wsm_rank, "promethee": prom_rank}
        tau_table = router.kendall_tau_table(rankings)
        assert tau_table.loc["wsm", "promethee"] == pytest.approx(1.0)


def test_topsis_dominator_ranks_first():
    """When one agent is best on every dimension, TOPSIS must rank it #1
    regardless of the weight vector."""
    matrix = _toy_matrix()  # alpha dominates beta, beta dominates gamma
    normalized = router.normalize(matrix)
    score = router.topsis(normalized, router.PRESETS["balanced"])
    assert score.index[0] == "alpha"


def test_pareto_optimal_includes_dominator():
    """A Pareto-dominant agent must appear in pareto_optimal output, and
    a strictly dominated agent must NOT."""
    matrix = _toy_matrix()  # alpha dominates beta dominates gamma
    pareto = router.pareto_optimal(matrix)
    # pareto_optimal returns the filtered DataFrame (only non-dominated rows).
    assert "alpha" in pareto.index
    assert "gamma" not in pareto.index


def test_pareto_2d_excludes_dominated_point():
    """A point that is dominated on both axes must NOT be Pareto-optimal."""
    matrix = pd.DataFrame(
        index=pd.Index(["best", "middle", "dominated"], name="agent"),
        data={
            "primary":     [1.0, 0.7, 0.5],   # higher better
            "D5_cost_usd": [0.0, 0.05, 0.10], # lower better
            "n_tasks_successful": [3, 3, 3],
            "n_tasks_total":      [3, 3, 3],
        },
    )
    mask = router.pareto_2d(
        matrix, x_col="D5_cost_usd", y_col="primary",
        x_lower_better=True, y_lower_better=False,
    )
    assert bool(mask.loc["best"])
    assert not bool(mask.loc["dominated"])


def test_kendall_tau_identity():
    """Kendall tau of any ranking with itself equals 1.0."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "d"])
    table = router.kendall_tau_table({"x": s, "y": s})
    assert table.loc["x", "y"] == pytest.approx(1.0)
    assert table.loc["x", "x"] == pytest.approx(1.0)


def test_recommend_returns_full_ranking_for_each_preset():
    """The end-to-end `recommend` call should produce a 9-agent ranking
    under every shipped preset, with no NaN scores."""
    for preset_name, weights in router.PRESETS.items():
        rec = router.recommend(weights)
        assert "rankings" in rec
        for method in ("wsm", "topsis", "promethee"):
            scores = rec["rankings"][method]
            assert len(scores) == 9, (
                f"{preset_name}/{method} returned {len(scores)} agents, expected 9"
            )
            assert not scores.isna().any(), (
                f"{preset_name}/{method} has NaN scores"
            )
