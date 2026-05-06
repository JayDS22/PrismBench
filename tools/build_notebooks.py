"""Generate prismbench.API.ipynb and prismbench.example.ipynb.

Produces two notebooks at the project root that match the
`tutorials/Autogen` reference layout the course expects.

Run with:
    .venv/bin/python tools/build_notebooks.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(*lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": list(lines),
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ============================================================================
# prismbench.API.ipynb
# ============================================================================

API_CELLS = [
    md(
        "# PrismBench API\n",
        "\n",
        "**PrismBench** is a benchmark and preference-aware routing framework for "
        "data science agents. This notebook tours the public Python API.\n",
        "\n",
        "Run order:\n",
        "\n",
        "1. Load the master scorecard summary (one row per *(agent, task)*).\n",
        "2. Build the per-agent routing matrix over the seven dimensions.\n",
        "3. Normalize via min-max so all dimensions live on `[0, 1]`.\n",
        "4. Score with three MCDM methods: WSM, TOPSIS, PROMETHEE-II.\n",
        "5. Compare method agreement with Kendall tau.\n",
        "6. Extract a 2D Pareto frontier.\n",
        "7. Run a sensitivity sweep on one weight axis.\n",
        "\n",
        "All functions are imported from `src.router`. Source:  "
        "[`src/router.py`](src/router.py).\n",
    ),
    md(
        "## 1. Load the scorecard\n",
        "\n",
        "`load_summary()` reads `results/master_scorecard_summary.csv`. That file "
        "is produced by `python -m src.scorecard` after a benchmark run; it has "
        "one row per *(agent, task)* with the mean across runs.\n",
    ),
    code(
        "import pandas as pd\n",
        "from src import router\n",
        "\n",
        "summary = router.load_summary()\n",
        "print(f'rows: {len(summary)}')\n",
        "print(f'agents: {sorted(summary[\"agent\"].unique())}')\n",
        "print(f'tasks:  {sorted(summary[\"task_id\"].unique())}')\n",
        "summary.head()\n",
    ),
    md(
        "## 2. Build the routing matrix\n",
        "\n",
        "`build_routing_matrix(summary, tasks=None)` aggregates the per-task rows "
        "into one row per agent with the seven routing dimensions populated. By "
        "default it filters to the three main benchmark tasks "
        "(`HD-PRED-01`, `TAXI-PRED-01`, `AR-PRED-01`).\n",
        "\n",
        "Dimensions (with direction):\n",
        "\n",
        "- `primary`     accuracy, higher is better\n",
        "- `D2_combined` code quality (pylint + LLM judge), higher is better\n",
        "- `D3_combined` explainability, higher is better\n",
        "- `D4_time_sec` wall-clock seconds, lower is better\n",
        "- `D5_cost_usd` API USD, lower is better\n",
        "- `D5_carbon_kg` kg CO2, lower is better\n",
        "- `D6_cv` coefficient of variation across runs, lower is better\n",
    ),
    code(
        "matrix = router.build_routing_matrix(summary)\n",
        "matrix.round(4)\n",
    ),
    md(
        "## 3. Normalize\n",
        "\n",
        "`normalize(matrix)` applies min-max normalization per dimension and "
        "flips the sign on lower-is-better dimensions, so every column ends up "
        "on `[0, 1]` with `1` = best.\n",
    ),
    code(
        "normalized = router.normalize(matrix)\n",
        "normalized.round(3)\n",
    ),
    md(
        "## 4. Score with three MCDM methods\n",
        "\n",
        "PrismBench implements three Multi-Criteria Decision Making methods:\n",
        "\n",
        "- **WSM** (Weighted Sum Model)  additive baseline.\n",
        "- **TOPSIS**  distance from ideal and anti-ideal points after weighted "
        "  normalization.\n",
        "- **PROMETHEE-II**  pairwise net flow with linear preference function.\n",
        "\n",
        "We use the `balanced` preset weight (uniform `1/7` on every dimension). "
        "Six other presets ship in `router.PRESETS` (`accuracy`, `frugal`, "
        "`quality`, `green`, `production`).\n",
    ),
    code(
        "weights = router.PRESETS['balanced']\n",
        "weights\n",
    ),
    code(
        "wsm_score       = router.wsm(normalized, weights)\n",
        "topsis_score    = router.topsis(normalized, weights)\n",
        "promethee_score = router.promethee(normalized, weights)\n",
        "\n",
        "ranks = pd.DataFrame({\n",
        "    'WSM':       wsm_score,\n",
        "    'TOPSIS':    topsis_score,\n",
        "    'PROMETHEE': promethee_score,\n",
        "}).round(3)\n",
        "ranks.sort_values('WSM', ascending=False)\n",
    ),
    md(
        "## 5. Method agreement: Kendall tau\n",
        "\n",
        "`kendall_tau_table` computes pairwise Kendall tau across the three "
        "rankings. Theoretical Proposition 1 in the report shows WSM and "
        "PROMETHEE-II are equivalent under min-max normalized scores with the "
        "linear preference function -- expect `tau = 1.0` for that pair.\n",
    ),
    code(
        "rankings = {\n",
        "    'wsm':       wsm_score,\n",
        "    'topsis':    topsis_score,\n",
        "    'promethee': promethee_score,\n",
        "}\n",
        "router.kendall_tau_table(rankings).round(3)\n",
    ),
    md(
        "## 6. Pareto frontier in 2D\n",
        "\n",
        "`pareto_2d` returns the non-dominated agents under a chosen 2D "
        "projection. We illustrate the accuracy-vs-cost projection: AutoGluon "
        "should be the unique non-dominated agent (highest D1 at zero per-run "
        "API cost).\n",
    ),
    code(
        "pareto = router.pareto_2d(\n",
        "    matrix,\n",
        "    x_col='D5_cost_usd',\n",
        "    y_col='primary',\n",
        "    x_lower_better=True,\n",
        "    y_lower_better=False,\n",
        ")\n",
        "print('Pareto-optimal in (cost, accuracy):', pareto)\n",
    ),
    code(
        "from pathlib import Path\n",
        "fig_path = Path('figures/api_pareto_demo.png')\n",
        "router.plot_pareto_2d(\n",
        "    matrix,\n",
        "    x_col='D5_cost_usd',\n",
        "    y_col='primary',\n",
        "    x_label='D5 cost (USD)  (lower = better)',\n",
        "    y_label='D1 normalized accuracy  (higher = better)',\n",
        "    title='Accuracy vs Cost (API demo)',\n",
        "    output_path=str(fig_path),\n",
        "    x_lower_better=True,\n",
        "    y_lower_better=False,\n",
        ")\n",
        "from IPython.display import Image\n",
        "Image(str(fig_path))\n",
    ),
    md(
        "## 7. Sensitivity sweep\n",
        "\n",
        "`sensitivity_sweep(target_dim, n_steps=21, method='wsm')` varies the "
        "target dimension's weight from `0` to `1` while distributing the "
        "complement uniformly across the remaining six dimensions. The point at "
        "which the top-ranked agent flips is a *breakpoint*; a routing "
        "recommendation is robust if no breakpoint sits near the user's nominal "
        "weight.\n",
    ),
    code(
        "sweep = router.sensitivity_sweep('D1', n_steps=21, method='wsm')\n",
        "sweep.head()\n",
    ),
    code(
        "breakpoints = router.find_top1_breakpoints(sweep)\n",
        "breakpoints\n",
    ),
    code(
        "fig_path = Path('figures/api_sensitivity_demo.png')\n",
        "router.plot_sensitivity(sweep, target_dim='D1', output_path=str(fig_path), method='wsm')\n",
        "Image(str(fig_path))\n",
    ),
    md(
        "## 8. One-shot: `recommend(weights)`\n",
        "\n",
        "If you only need the headline answer, `recommend` runs all three "
        "methods and returns the top agent under each:\n",
    ),
    code(
        "for preset_name in ['balanced', 'accuracy', 'frugal', 'green']:\n",
        "    rec = router.recommend(router.PRESETS[preset_name])\n",
        "    tops = {m: list(s.index)[0] for m, s in rec['rankings'].items()}\n",
        "    print(f'{preset_name:>10}: {tops}')\n",
    ),
    md(
        "## Where to go next\n",
        "\n",
        "- [`prismbench.example.ipynb`](prismbench.example.ipynb)  end-to-end run "
        "of one agent on one task.\n",
        "- [`report.pdf`](report.pdf)  full empirical study with statistical "
        "validation, multi-protocol analysis, and Theoretical Proposition 1.\n",
        "- [`results/master_scorecard.csv`](results/master_scorecard.csv)  the "
        "raw cell-level data backing every figure and number in the report.\n",
        "- [`src/router.py`](src/router.py)  source for everything imported here.\n",
    ),
]


# ============================================================================
# prismbench.example.ipynb
# ============================================================================

EXAMPLE_CELLS = [
    md(
        "# PrismBench Example\n",
        "\n",
        "End-to-end walkthrough of how a single cell of the PrismBench "
        "benchmark is produced. We trace one *(agent, task, run)* triple from "
        "configuration to scorecard, then show how 165 such cells aggregate "
        "into a routing recommendation.\n",
        "\n",
        "Steps:\n",
        "\n",
        "1. Open the task config and the prompt every agent sees.\n",
        "2. Inspect the dataset that gets staged into the agent workspace.\n",
        "3. Open one cached *(agent, task, run)* result.\n",
        "4. Aggregate three runs into mean and CV.\n",
        "5. Connect the per-cell results to the master scorecard.\n",
        "6. Feed the master scorecard to the router for a recommendation.\n",
        "7. (Optional) Re-execute one agent live, if API keys are present.\n",
    ),
    md(
        "## 1. The task config\n",
        "\n",
        "All task definitions live in [`configs/tasks.yaml`](configs/tasks.yaml). "
        "Each task carries its own prompt, dataset binding, primary metric, "
        "and split seed. We pick `HD-PRED-01` (heart-disease binary "
        "classification, 303 rows) for this walkthrough because every agent "
        "succeeds on it.\n",
    ),
    code(
        "from src.task_runner import load_tasks_config\n",
        "\n",
        "tasks = load_tasks_config()\n",
        "task = tasks['HD-PRED-01']\n",
        "print('task_id:       ', task.get('task_id', 'HD-PRED-01'))\n",
        "print('dataset:       ', task['dataset'])\n",
        "print('task_type:     ', task['task_type'])\n",
        "print('primary metric:', task['primary_metric'])\n",
        "print('split seed:    ', task.get('split_seed'))\n",
    ),
    code(
        "# The exact prompt every agent receives, verbatim. Section 3.3 of the\n",
        "# report calls this the *same-prompt rule* (Protocol A).\n",
        "print(task['prompt'])\n",
    ),
    md(
        "## 2. The dataset\n",
        "\n",
        "Before each run, `task_runner` stages the dataset into the agent's "
        "per-run workspace. The agent reads the file by name -- it never has "
        "to resolve paths.\n",
    ),
    code(
        "import pandas as pd\n",
        "df = pd.read_csv('data/raw/heart_disease.csv')\n",
        "print(f'shape: {df.shape}')\n",
        "df.head()\n",
    ),
    md(
        "## 3. Open one cached result\n",
        "\n",
        "Every agent run lands in `results/{agent}/{task}/run_{n}/` and "
        "contains:\n",
        "\n",
        "- `result.json`  the raw output dict (predictions, code, tokens, etc.)\n",
        "- `scorecard.json`  the six-dimension scoring of that run\n",
        "- `workspace/`  the staged dataset and any agent-generated files\n",
        "\n",
        "We open `autogluon` on `HD-PRED-01`, run 1 below.\n",
    ),
    code(
        "import json\n",
        "from pathlib import Path\n",
        "\n",
        "run_dir = Path('results/autogluon/HD-PRED-01/run_1')\n",
        "print('files in run dir:')\n",
        "for p in sorted(run_dir.iterdir()):\n",
        "    print(' ', p.name)\n",
    ),
    code(
        "scorecard = json.loads((run_dir / 'scorecard.json').read_text())\n",
        "print(json.dumps({\n",
        "    'agent':   scorecard['agent'],\n",
        "    'task_id': scorecard['task_id'],\n",
        "    'run_id':  scorecard['run_id'],\n",
        "    'D1':      scorecard.get('D1_accuracy'),\n",
        "    'D2':      scorecard.get('D2_code_quality'),\n",
        "    'D3':      scorecard.get('D3_explainability'),\n",
        "    'D4':      scorecard.get('D4_speed'),\n",
        "    'D5':      scorecard.get('D5_cost'),\n",
        "    'D6':      scorecard.get('D6_robustness'),\n",
        "}, indent=2, default=str))\n",
    ),
    md(
        "## 4. Aggregate three runs\n",
        "\n",
        "Each (agent, task) cell is run three times. The robustness dimension "
        "`D6` is the coefficient of variation of `D1` across those runs -- a "
        "consistent agent has low CV.\n",
    ),
    code(
        "import numpy as np\n",
        "\n",
        "f1_scores = []\n",
        "for n in (1, 2, 3):\n",
        "    sc = json.loads((Path(f'results/autogluon/HD-PRED-01/run_{n}/scorecard.json')).read_text())\n",
        "    d1 = sc.get('D1_accuracy', {}) or {}\n",
        "    f1 = d1.get('f1_weighted')\n",
        "    f1_scores.append(f1)\n",
        "    print(f'run {n}: F1 = {f1}')\n",
        "\n",
        "f1_arr = np.array([s for s in f1_scores if s is not None], dtype=float)\n",
        "print(f'\\nmean F1: {f1_arr.mean():.4f}')\n",
        "print(f'std F1:  {f1_arr.std(ddof=0):.4f}')\n",
        "if f1_arr.mean() > 0:\n",
        "    print(f'CV (D6): {f1_arr.std(ddof=0) / f1_arr.mean():.4f}')\n",
    ),
    md(
        "## 5. Connect to the master scorecard\n",
        "\n",
        "`src.scorecard.build_scorecard()` walks `results/` and produces "
        "`results/master_scorecard.csv` (one row per run) and "
        "`results/master_scorecard_summary.csv` (one row per *(agent, task)* "
        "with the mean across runs). The summary CSV is what the router "
        "consumes.\n",
    ),
    code(
        "master = pd.read_csv('results/master_scorecard.csv')\n",
        "print(f'total rows: {len(master)}')\n",
        "\n",
        "ag = master[(master.agent == 'autogluon') & (master.task_id == 'HD-PRED-01')]\n",
        "ag[['agent', 'task_id', 'run_id', 'D1_f1', 'D4_time_sec', 'D5_cost_usd', 'D5_carbon_kg']]\n",
    ),
    md(
        "## 6. Feed the router\n",
        "\n",
        "The router ingests the summary CSV and returns a top-1 agent under "
        "each MCDM method, given a weight vector. We try the `accuracy` "
        "preset (60% on D1) and the `frugal` preset (30% cost, 20% carbon).\n",
    ),
    code(
        "from src import router\n",
        "\n",
        "for preset_name in ['balanced', 'accuracy', 'frugal']:\n",
        "    rec = router.recommend(router.PRESETS[preset_name])\n",
        "    print(f'\\n=== preset: {preset_name} ===')\n",
        "    for method, scores in rec['rankings'].items():\n",
        "        top3 = list(scores.index[:3])\n",
        "        print(f'  {method:>10}: {top3}')\n",
    ),
    md(
        "## 7. (Optional) Live run\n",
        "\n",
        "The cells above use cached results so this notebook executes in seconds. "
        "If you want to see the loop run live, the cell below executes "
        "`autogluon` on a *fresh* run. It takes about 90-120 seconds and writes "
        "to `results/autogluon/HD-PRED-01/run_99/`.\n",
        "\n",
        "Skip this cell during grading review if you don't want to wait.\n",
    ),
    code(
        "# Uncomment to run live. Requires the autogluon Python package; no API\n",
        "# key needed.\n",
        "#\n",
        "# from src.task_runner import run_single\n",
        "# result = run_single('autogluon', 'HD-PRED-01', run_id=99)\n",
        "# print({k: result[k] for k in ('agent_id', 'task_id', 'wall_clock_sec', 'error')})\n",
    ),
    md(
        "## Where to go next\n",
        "\n",
        "- [`prismbench.API.ipynb`](prismbench.API.ipynb)  pure-API tour of the "
        "router.\n",
        "- [`report.pdf`](report.pdf)  full study, including the multi-protocol "
        "analysis and Theoretical Proposition 1.\n",
        "- `python -m src.run_benchmark --pilot`  three agents on two tasks, "
        "single run; ten minutes end to end.\n",
        "- `python -m src.run_benchmark`  full 81-cell main benchmark; an hour "
        "or so, depending on API rate limits.\n",
    ),
]


def main():
    api_path = ROOT / "prismbench.API.ipynb"
    example_path = ROOT / "prismbench.example.ipynb"
    api_path.write_text(json.dumps(notebook(API_CELLS), indent=1))
    example_path.write_text(json.dumps(notebook(EXAMPLE_CELLS), indent=1))
    print(f"wrote {api_path}")
    print(f"wrote {example_path}")


if __name__ == "__main__":
    main()
