# Smoke Test Baseline — 2026-05-05

First end-to-end smoke pass with the full agent roster after the gemini → chatgpt_ada swap.

- **Date:** 2026-05-05
- **HEAD at run time:** 9dda57a (uncommitted changes on top — see commit plan)
- **Branch:** UmdTask403_DATA605_Spring2026_Comparison_of_Data_Science_Agents
- **Command:** `python -m src.smoke_test`
- **Smoke task:** trivial heart-disease classifier (`SMOKE-TEST` task config in `src/smoke_test.py`)

## Result: 9/9 PASS

| Agent | Status | Time | Code | Preds | Tokens |
|---|---|---|---|---|---|
| autogluon       | PASS | 127.0s  | N | Y | 0 |
| pycaret         | PASS |   9.4s  | N | Y | 0 |
| chatgpt_ada     | PASS |  24.4s  | Y | N | 569 |
| claude_code     | PASS |  22.3s  | Y | N | 266 |
| claude_api_raw  | PASS |   3.5s  | Y | N | 331 |
| autogen         | PASS |  27.1s  | Y | N | 0 |
| smolagents      | PASS | 576.8s  | N | N | 0 |
| langgraph       | PASS |   5.2s  | Y | N | 312 |
| pandasai        | PASS |   3.9s  | N | N | 0 |

Total wall-clock: ~13 min (smolagents accounts for ~10 min by itself).

## Notes / known gaps

- **smolagents is 200–300× slower than its peers.** Its iterative search-plan-repair loop multiplies GPT-4o calls. A full 9 × 20 × 3 matrix at this rate ≈ 50 hours of just-smolagents. Cap with `max_steps` or a wall-clock timeout before running the full benchmark.
- **`autogen`, `smolagents`, `pandasai` report `tokens=0` despite making API calls.** Their wrappers don't pull `usage` out of SDK responses. D5 (cost) will report $0 for all three until fixed.
- **Code/Preds columns are mutually exclusive.** AutoML agents (autogluon/pycaret) emit predictions but no source. LLM agents emit source but no predictions in the smoke prompt (smoke prompt doesn't ask for `predictions.csv`). Both should appear once running the real `tasks.yaml` prompts.
