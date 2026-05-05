"""
Runs any agent on any task, captures output, scores on 6 dimensions.

Usage:
    python -m src.task_runner --agent autogluon --task HD-PRED-01 --runs 1
    python -m src.task_runner --agent claude_api_raw --task HD-PRED-01 --runs 3
    python -m src.task_runner --agent all --task all --runs 3
"""
import sys
import argparse
import importlib
import shutil
from pathlib import Path

from src.utils import (
    log, save_json, load_json, load_tasks_config, load_agents_config,
    load_datasets_config, get_result_dir, make_result, Timer,
    DATA_RAW, DATA_ADVERSARIAL, AGENTS_DIR, PROJECT_ROOT,
)
from src.evaluator import evaluate_result


def load_agent_runner(agent_id):
    """Dynamically import agents/{agent_id}/run_task.py and return its run() function."""
    agent_module_path = AGENTS_DIR / agent_id / "run_task.py"
    if not agent_module_path.exists():
        raise FileNotFoundError(f"Agent wrapper not found: {agent_module_path}")
    spec = importlib.util.spec_from_file_location(f"agents.{agent_id}.run_task", agent_module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run"):
        raise AttributeError(f"agents/{agent_id}/run_task.py must define a run() function")
    return mod.run


def _copy_dataset_to_workspace(task_cfg, work_dir):
    """
    Copy the task's dataset file into work_dir so the agent has a clean workspace.
    Logs a warning if the file cannot be located — agents should fail loudly on missing data.
    """
    dataset_name = task_cfg["dataset"]
    adversarial = task_cfg.get("adversarial", False)
    data_dir = DATA_ADVERSARIAL if adversarial else DATA_RAW

    try:
        dataset_cfg = load_datasets_config()
    except Exception as e:
        log(f"WARNING: could not load datasets config: {e}")
        return

    if dataset_name not in dataset_cfg:
        log(f"WARNING: dataset '{dataset_name}' not found in datasets.yaml")
        return

    local_path = dataset_cfg[dataset_name].get("local_path", "")
    filename = Path(local_path).name

    candidates = [
        PROJECT_ROOT / local_path,
        PROJECT_ROOT / "data" / "raw" / filename,
        data_dir / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            dest = work_dir / candidate.name
            if not dest.exists():
                shutil.copy2(candidate, dest)
            return

    log(f"WARNING: data file for '{dataset_name}' not found in any expected location")


def run_single(agent_id, task_id, run_id=1):
    """
    Execute one (agent, task, run) triple and return the scorecard dict.

    Flow:
        1. Load task + agent config
        2. Prepare workspace and copy dataset
        3. Dispatch to agent wrapper
        4. Score on 6 dimensions
        5. Save result.json + scorecard.json
    """
    tasks  = load_tasks_config()
    agents = load_agents_config()

    if task_id not in tasks:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(tasks.keys())}")
    if agent_id not in agents:
        raise ValueError(f"Unknown agent: {agent_id}. Available: {list(agents.keys())}")

    task_cfg  = tasks[task_id]
    agent_cfg = agents[agent_id]

    task_modality = task_cfg.get("modality", "tabular")
    supported     = agent_cfg.get("supports", [])
    if supported and task_modality not in supported:
        log(f"WARNING: {agent_id} does not declare support for modality '{task_modality}', running anyway")

    out_dir  = get_result_dir(agent_id, task_id, run_id)
    work_dir = out_dir / "workspace"
    work_dir.mkdir(exist_ok=True)

    _copy_dataset_to_workspace(task_cfg, work_dir)

    try:
        agent_run = load_agent_runner(agent_id)
    except (FileNotFoundError, AttributeError) as e:
        log(f"ERROR: cannot load agent {agent_id}: {e}")
        result = make_result(agent_id, task_id, run_id, error=str(e))
        save_json(result, out_dir / "result.json")
        return {"error": str(e)}

    log(f"[task_runner] {agent_id} × {task_id} run {run_id}")

    # CodeCarbon: measure local-machine energy during the agent run. Cloud
    # agents that call remote APIs will register tiny numbers here (only
    # the orchestration overhead) — that's fine; cost_tracker uses the
    # token-based estimate for cloud-side compute.
    tracker = None
    try:
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(
            measure_power_secs=2,
            save_to_file=False,
            log_level="error",
            tracking_mode="process",
        )
        tracker.start()
    except Exception as e:
        log(f"[task_runner] CodeCarbon unavailable: {e}")
        tracker = None

    with Timer() as t:
        try:
            result = agent_run(
                prompt=task_cfg.get("prompt", ""),
                task_config=task_cfg,
                work_dir=str(work_dir),
                output_dir=str(out_dir),
            )
        except Exception as e:
            log(f"ERROR: {agent_id} raised an unhandled exception: {e}")
            result = make_result(agent_id, task_id, run_id, error=str(e))

    measured_kg = None
    if tracker is not None:
        try:
            measured_kg = tracker.stop()
        except Exception as e:
            log(f"[task_runner] CodeCarbon stop failed: {e}")

    result.setdefault("agent", agent_id)
    result.setdefault("task_id", task_id)
    result.setdefault("run_id", run_id)
    result.setdefault("wall_clock_sec", t.elapsed)
    result["carbon_kg_measured"] = measured_kg

    save_json(result, out_dir / "result.json")

    scorecard = evaluate_result(result, task_cfg)
    save_json(scorecard, out_dir / "scorecard.json")

    log(f"[task_runner] done {t.elapsed:.1f}s | D1={scorecard.get('D1_accuracy')} | D5={scorecard.get('D5_cost')}")

    return scorecard


def run_batch(agent_id, task_id, n_runs=3):
    """Run one (agent, task) pair n_runs times and return all scorecards."""
    return [run_single(agent_id, task_id, run_id) for run_id in range(1, n_runs + 1)]


def run_all(agent_ids, task_ids, n_runs=3):
    """Run all (agent, task) combinations and return all scorecards."""
    total = len(agent_ids) * len(task_ids) * n_runs
    log(f"[task_runner] starting {len(agent_ids)} agents × {len(task_ids)} tasks × {n_runs} runs = {total} total")
    all_scorecards = []
    for agent_id in agent_ids:
        for task_id in task_ids:
            all_scorecards.extend(run_batch(agent_id, task_id, n_runs))
    log(f"[task_runner] complete — {len(all_scorecards)} scorecards generated")
    return all_scorecards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DS-RouteBench Task Runner")
    parser.add_argument("--agent", required=True, help="Agent ID or 'all'")
    parser.add_argument("--task",  required=True, help="Task ID or 'all'")
    parser.add_argument("--runs",  type=int, default=1, help="Number of runs per (agent, task)")
    args = parser.parse_args()

    tasks  = load_tasks_config()
    agents = load_agents_config()

    agent_ids = list(agents.keys()) if args.agent == "all" else [args.agent]
    task_ids  = list(tasks.keys())  if args.task  == "all" else [args.task]

    if len(agent_ids) == 1 and len(task_ids) == 1:
        scorecards = run_batch(agent_ids[0], task_ids[0], args.runs)
    else:
        scorecards = run_all(agent_ids, task_ids, args.runs)

    print(f"\n{'='*60}")
    print(f"Completed : {len(scorecards)} scorecards")
    print(f"Results   : results/")
    print(f"{'='*60}")