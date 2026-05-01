"""
Run all (or selected) agents × tasks × runs and generate scorecard.

Usage:
    python -m src.run_benchmark --agents all --tasks all --runs 3
    python -m src.run_benchmark --agents autogluon,claude_api_raw --tasks HD-PRED-01,TAXI-PRED-01 --runs 3
    python -m src.run_benchmark --pilot   # 3 agents × 2 tasks × 1 run
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import log, load_tasks_config, load_agents_config
from src.task_runner import run_all
from src.scorecard import build_scorecard, print_agent_rankings

PILOT_AGENTS = ["autogluon", "pycaret", "claude_api_raw"]
PILOT_TASKS  = ["HD-PRED-01", "TAXI-PRED-01"]


def main():
    parser = argparse.ArgumentParser(description="DS-RouteBench Benchmark Runner")
    parser.add_argument("--agents", type=str, default="all", help="Comma-separated agent IDs or 'all'")
    parser.add_argument("--tasks",  type=str, default="all", help="Comma-separated task IDs or 'all'")
    parser.add_argument("--runs",   type=int, default=3,     help="Number of runs per (agent, task)")
    parser.add_argument("--pilot",  action="store_true",     help="Quick pilot: 3 agents × 2 tasks × 1 run")
    args = parser.parse_args()

    tasks_cfg  = load_tasks_config()
    agents_cfg = load_agents_config()

    if args.pilot:
        agent_ids = [a for a in PILOT_AGENTS if a in agents_cfg]
        task_ids  = [t for t in PILOT_TASKS  if t in tasks_cfg]
        n_runs    = 1
    else:
        agent_ids = list(agents_cfg.keys()) if args.agents == "all" else args.agents.split(",")
        task_ids  = list(tasks_cfg.keys())  if args.tasks  == "all" else args.tasks.split(",")
        n_runs    = args.runs

    unknown_agents = [a for a in agent_ids if a not in agents_cfg]
    unknown_tasks  = [t for t in task_ids  if t not in tasks_cfg]
    for a in unknown_agents:
        log(f"WARNING: unknown agent '{a}', skipping")
    for t in unknown_tasks:
        log(f"WARNING: unknown task '{t}', skipping")

    agent_ids = [a for a in agent_ids if a in agents_cfg]
    task_ids  = [t for t in task_ids  if t in tasks_cfg]

    if not agent_ids or not task_ids:
        log("ERROR: no valid agents or tasks to run. Exiting.")
        sys.exit(1)

    total = len(agent_ids) * len(task_ids) * n_runs
    log(
        f"\n{'='*60}\n"
        f"  Agents : {agent_ids}\n"
        f"  Tasks  : {task_ids}\n"
        f"  Runs   : {n_runs}\n"
        f"  Total  : {total} experiments\n"
        f"{'='*60}"
    )

    all_scorecards = run_all(agent_ids, task_ids, n_runs)

    log("Generating master scorecard...")
    summary = build_scorecard()
    print_agent_rankings(summary)
    log(f"Benchmark complete — {len(all_scorecards)} scorecards written to results/")


if __name__ == "__main__":
    main()