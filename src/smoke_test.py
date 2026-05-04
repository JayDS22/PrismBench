"""
Agent Smoke Test
Verify each agent can be loaded and run on a minimal task.

Usage:
    python src/smoke_test.py
    python src/smoke_test.py --agent autogluon
"""
import sys
import time
import shutil
import tempfile
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import log, load_agents_config, AGENTS_DIR


SMOKE_TASK_CONFIG = {
    "task_id": "SMOKE-TEST",
    "dataset": "heart_disease",
    "analytics_level": "predictive",
    "modality": "tabular",
    "task_type": "classification",
    "primary_metric": "f1",
    "split_seed": 42,
    "test_size": 0.2,
    "adversarial": False,
    "prompt": """You are given heart_disease.csv. Build a simple classifier for the 'target' column.
Split 80/20 (random_state=42). Report accuracy. Save code as solution.py.""",
}


def smoke_test_agent(agent_id):
    """Test if an agent can be loaded and returns a result dict."""
    log(f"  Testing {agent_id}...")

    # Check if wrapper exists
    wrapper_path = AGENTS_DIR / agent_id / "run_task.py"
    if not wrapper_path.exists():
        return {"agent": agent_id, "status": "MISSING", "error": "run_task.py not found", "time": 0}

    # Try to import
    try:
        from src.task_runner import load_agent_runner
        runner = load_agent_runner(agent_id)
    except Exception as e:
        return {"agent": agent_id, "status": "IMPORT FAIL", "error": str(e)[:100], "time": 0}

    # Check if data exists
    from src.utils import DATA_RAW
    data_file = DATA_RAW / "heart_disease.csv"
    if not data_file.exists():
        return {"agent": agent_id, "status": "NO DATA", "error": "Run data_loader.py --download-all first", "time": 0}

    # Run agent on test task
    work_dir = tempfile.mkdtemp(prefix=f"smoke_{agent_id}_")
    out_dir = tempfile.mkdtemp(prefix=f"smoke_out_{agent_id}_")

    # Copy data to work dir
    shutil.copy2(data_file, Path(work_dir) / "heart_disease.csv")

    start = time.perf_counter()
    try:
        result = runner(
            prompt=SMOKE_TASK_CONFIG["prompt"],
            task_config=SMOKE_TASK_CONFIG,
            work_dir=work_dir,
            output_dir=out_dir,
        )
        elapsed = time.perf_counter() - start

        if result.get("error"):
            return {"agent": agent_id, "status": "ERROR", "error": result["error"][:100], "time": round(elapsed, 1)}
        else:
            has_code = "Y" if result.get("generated_code") else "N"
            has_preds = "Y" if result.get("predictions") else "N"
            return {
                "agent": agent_id,
                "status": "PASS",
                "time": round(elapsed, 1),
                "code": has_code,
                "preds": has_preds,
                "tokens": result.get("tokens_used", 0),
                "error": None,
            }

    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"agent": agent_id, "status": "CRASH", "error": str(e)[:100], "time": round(elapsed, 1)}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


def run_all_smoke_tests(agent_ids=None):
    """Run smoke tests for all or specified agents."""
    agents = load_agents_config()
    if agent_ids:
        test_ids = agent_ids
    else:
        test_ids = list(agents.keys())

    log(f"\n{'='*60}")
    log(f"Smoke Testing {len(test_ids)} agents")
    log(f"{'='*60}\n")

    results = []
    for agent_id in test_ids:
        r = smoke_test_agent(agent_id)
        results.append(r)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Agent':<20} {'Status':<15} {'Time':>8} {'Code':>6} {'Preds':>6} {'Tokens':>8} {'Error'}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r['agent']:<20} {r['status']:<15} {r.get('time', 0):>7.1f}s {r.get('code', '-'):>6} {r.get('preds', '-'):>6} {r.get('tokens', 0):>8} {r.get('error', '') or ''}")
    print(f"{'='*80}")

    passed = sum(1 for r in results if "PASS" in r["status"])
    print(f"\n  {passed}/{len(results)} agents passed smoke test.\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke Test")
    parser.add_argument("--agent", type=str, help="Test a specific agent")
    args = parser.parse_args()

    if args.agent:
        run_all_smoke_tests([args.agent])
    else:
        run_all_smoke_tests()
