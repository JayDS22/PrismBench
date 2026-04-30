"""
Agent: PyCaret
Category: AutoML / Low-code (local, no tokens)
"""
import os
import sys
import time
import json
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import make_result, log


def run(prompt, task_config, work_dir, output_dir):
    """Run PyCaret on a task. Not prompt-based — uses task_config."""
    task_type = task_config.get("task_type", "classification")
    dataset_name = task_config.get("dataset")

    if task_type in ("descriptive", "eda", "prescriptive", "explainability", "end_to_end"):
        return make_result("pycaret", task_config.get("task_id", "?"),
                           error=f"PyCaret does not support task_type='{task_type}'.")

    # Load data
    data_files = [f for f in os.listdir(work_dir) if f.endswith((".csv", ".parquet"))]
    if not data_files:
        return make_result("pycaret", task_config.get("task_id", "?"), error="No data files")

    data_file = os.path.join(work_dir, data_files[0])
    df = pd.read_csv(data_file) if data_file.endswith(".csv") else pd.read_parquet(data_file)

    from src.utils import load_datasets_config
    ds_cfg = load_datasets_config().get(dataset_name, {})
    target_col = ds_cfg.get("target_col", df.columns[-1])

    seed = task_config.get("split_seed", 42)

    start = time.perf_counter()

    if task_type == "classification":
        from pycaret.classification import setup, compare_models, predict_model, pull
        setup(data=df, target=target_col, session_id=seed, verbose=False, html=False)
        best = compare_models(n_select=1, verbose=False)
        comparison = pull()
        preds_df = predict_model(best, verbose=False)
        y_pred = preds_df["prediction_label"].tolist()
        y_true = preds_df[target_col].tolist()
        y_prob = preds_df.get("prediction_score", pd.Series()).tolist() or None

    elif task_type in ("regression", "forecasting"):
        from pycaret.regression import setup, compare_models, predict_model, pull
        setup(data=df, target=target_col, session_id=seed, verbose=False, html=False)
        best = compare_models(n_select=1, verbose=False)
        comparison = pull()
        preds_df = predict_model(best, verbose=False)
        y_pred = preds_df["prediction_label"].tolist()
        y_true = preds_df[target_col].tolist()
        y_prob = None
    else:
        return make_result("pycaret", task_config.get("task_id", "?"),
                           error=f"Unsupported task_type: {task_type}")

    elapsed = time.perf_counter() - start

    return make_result(
        agent_id="pycaret",
        task_id=task_config.get("task_id", "?"),
        predictions=y_pred,
        y_true=y_true,
        y_prob=y_prob,
        wall_clock_sec=elapsed,
        cost_usd=0.0,
        tokens_used=0,
        generated_code=None,
        raw_output=json.dumps({"comparison_table": comparison.to_dict() if comparison is not None else {}}, default=str),
    )
