"""One-shot smoke test for the Docker container.

Runs four checks:
  1. Import prismbench_utils.
  2. Build routing matrix and identify top agent.
  3. Open one cached scorecard.json.
  4. Execute both notebooks end-to-end via nbclient.

Print "PASS"/"FAIL" lines so a grep can summarize.
"""
import sys
import traceback
from pathlib import Path

# Project root on sys.path so prismbench_utils + src/ resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def step(label, fn):
    try:
        fn()
        print(f"PASS  {label}")
    except Exception as e:
        print(f"FAIL  {label}: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)


def check_import():
    import prismbench_utils as pbu  # noqa: F401


def check_router():
    import prismbench_utils as pbu

    m = pbu.build_routing_matrix(pbu.load_summary())
    assert m.shape[0] == 9, f"expected 9 agents, got {m.shape[0]}"
    top = pbu.top_agent(pbu.PRESETS["balanced"])
    assert top in m.index, f"top agent {top!r} not in matrix"
    print(f"      top under balanced: {top}")


def check_scorecard():
    import prismbench_utils as pbu

    sc = pbu.load_run_scorecard("autogluon", "HD-PRED-01", 1)
    assert sc["agent"] == "autogluon"
    f1 = sc["D1_accuracy"]["f1_weighted"]
    print(f"      autogluon HD-PRED-01 run_1 F1: {f1}")


def check_notebooks():
    import subprocess

    for nb_path in ("prismbench.API.ipynb", "prismbench.example.ipynb"):
        # Use jupyter nbconvert (ships with the base Jupyter install) to
        # execute every cell. --to notebook + --output to /tmp keeps the
        # original file unchanged while still failing on any cell error.
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=120",
                "--output", f"/tmp/{nb_path}",
                nb_path,
            ],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{nb_path} failed: {result.stderr[-500:]}"
            )
        print(f"      {nb_path}: executed cleanly")


if __name__ == "__main__":
    step("import prismbench_utils", check_import)
    step("router pipeline",         check_router)
    step("load run scorecard",      check_scorecard)
    step("execute notebooks",       check_notebooks)
    print("ALL CHECKS PASSED")
