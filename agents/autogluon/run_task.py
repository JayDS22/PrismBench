"""
Agent: AutoGluon
Category: AutoML (local, no LLM, no tokens)
"""
import os
import sys
import time
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import make_result, log


def run(prompt, task_config, work_dir, output_dir):
    """
    Run AutoGluon on a task.
    AutoGluon is NOT prompt-based — it uses the task_config directly.
    """
    from autogluon.tabular import TabularPredictor

    task_type = task_config.get("task_type", "classification")
    dataset_name = task_config.get("dataset")
    target_col = None

    # Load the data from work_dir
    data_files = [f for f in os.listdir(work_dir) if f.endswith((".csv", ".parquet"))]
    if not data_files:
        return make_result("autogluon", task_config.get("task_id", "?"), error="No data files in work_dir")

    data_file = os.path.join(work_dir, data_files[0])
    if data_file.endswith(".csv"):
        df = pd.read_csv(data_file)
    else:
        df = pd.read_parquet(data_file)

    # Determine target column from task config or dataset config
    from src.utils import load_datasets_config
    ds_cfg = load_datasets_config().get(dataset_name, {})
    target_col = ds_cfg.get("target_col", df.columns[-1])

    if target_col not in df.columns:
        return make_result("autogluon", task_config.get("task_id", "?"),
                           error=f"Target '{target_col}' not in columns: {list(df.columns)}")

    # Handle non-predictive tasks
    if task_type in ("descriptive", "eda", "prescriptive", "explainability", "end_to_end"):
        return make_result(
            "autogluon", task_config.get("task_id", "?"),
            error=f"AutoGluon does not support task_type='{task_type}'. Only classification/regression.",
        )

    # Split
    seed = task_config.get("split_seed", 42)
    test_size = task_config.get("test_size", 0.2)
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=seed)

    # Train
    predictor_path = os.path.join(output_dir, "autogluon_model")
    start = time.perf_counter()
    predictor = TabularPredictor(
        label=target_col,
        path=predictor_path,
        verbosity=0,
    ).fit(
        df_train,
        time_limit=120,
        presets="best_quality",
    )

    # Predict
    y_pred = predictor.predict(df_test).tolist()
    y_true = df_test[target_col].tolist()

    # Probabilities (for classification)
    y_prob = None
    if task_type == "classification":
        try:
            probs = predictor.predict_proba(df_test)
            if probs.shape[1] == 2:
                y_prob = probs.iloc[:, 1].tolist()
            else:
                y_prob = probs.values.tolist()
        except Exception:
            y_prob = None

    elapsed = time.perf_counter() - start

    # Leaderboard
    try:
        lb = predictor.leaderboard(df_test, silent=True)
        leaderboard = lb.head(5).to_dict(orient="records")
    except Exception:
        leaderboard = []

    return make_result(
        agent_id="autogluon",
        task_id=task_config.get("task_id", "?"),
        predictions=y_pred,
        y_true=y_true,
        y_prob=y_prob,
        wall_clock_sec=elapsed,
        cost_usd=0.0,
        tokens_used=0,
        generated_code=None,  # AutoGluon doesn't generate code
        raw_output=json.dumps({"leaderboard": leaderboard}, default=str),
    )
