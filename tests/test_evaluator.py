"""Unit tests for src.evaluator scoring primitives."""
import pytest

from src import evaluator


# ---------------------------------------------------------------------------
# D1 accuracy
# ---------------------------------------------------------------------------

def test_score_accuracy_classification_perfect():
    """Identical y_true and y_pred give F1 = 1.0 and accuracy = 1.0."""
    y = [0, 1, 0, 1, 1, 0]
    result = evaluator.score_accuracy("classification", y_true=y, y_pred=y)

    assert result["type"] == "classification"
    assert result["f1_weighted"] == pytest.approx(1.0)
    assert result["accuracy"] == pytest.approx(1.0)


def test_score_accuracy_classification_chance():
    """Inverted predictions on a balanced binary task give accuracy 0.0."""
    y_true = [0, 1, 0, 1]
    y_pred = [1, 0, 1, 0]
    result = evaluator.score_accuracy("classification", y_true=y_true, y_pred=y_pred)

    assert result["accuracy"] == pytest.approx(0.0)


def test_score_accuracy_regression_zero_error():
    """Perfect regression predictions give RMSE = 0 and R^2 = 1."""
    y = [1.0, 2.0, 3.0, 4.0]
    result = evaluator.score_accuracy("regression", y_true=y, y_pred=y)

    assert result["type"] == "regression"
    assert result["rmse"] == pytest.approx(0.0)
    assert result["r2"] == pytest.approx(1.0)


def test_score_accuracy_handles_string_labels():
    """Categorical string labels (e.g. NLP sentiment) round-trip cleanly."""
    y = ["positive", "negative", "positive", "neutral"]
    result = evaluator.score_accuracy("classification", y_true=y, y_pred=y)

    assert result["f1_weighted"] == pytest.approx(1.0)


def test_score_accuracy_missing_predictions_returns_error():
    """When predictions are absent the scorer returns a structured error
    dict instead of crashing the run."""
    result = evaluator.score_accuracy("classification", y_true=[0, 1], y_pred=None)
    assert "error" in result


# ---------------------------------------------------------------------------
# D6 robustness (CV across runs)
# ---------------------------------------------------------------------------

def test_score_robustness_zero_variance():
    """Identical scores across runs have CV = 0 (perfectly robust)."""
    result = evaluator.score_robustness([0.9, 0.9, 0.9])
    assert result["cv"] == pytest.approx(0.0)
    assert result["n_runs"] == 3


def test_score_robustness_basic_cv():
    """CV is std/mean for a known small sample."""
    result = evaluator.score_robustness([0.8, 1.0, 0.9])
    # mean = 0.9, population std = sqrt(2/300) ~= 0.0816, CV ~= 0.0907.
    assert result["cv"] == pytest.approx(0.0816 / 0.9, abs=0.01)


# ---------------------------------------------------------------------------
# D5 cost / carbon helpers (ensure config-driven pricing flows through)
# ---------------------------------------------------------------------------

def test_calculate_cost_zero_for_local_agent():
    """Local agents (autogluon, pycaret) have zero per-token pricing."""
    cost = evaluator.calculate_cost("autogluon", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(0.0)


def test_calculate_cost_scales_linearly_with_tokens():
    """Doubling tokens doubles the API cost for a priced agent."""
    a = evaluator.calculate_cost("claude_api_raw", input_tokens=1000, output_tokens=1000)
    b = evaluator.calculate_cost("claude_api_raw", input_tokens=2000, output_tokens=2000)
    if a > 0:  # only assert if pricing is configured
        assert b == pytest.approx(2 * a, rel=1e-6)
