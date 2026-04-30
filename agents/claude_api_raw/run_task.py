"""
Agent: Claude API (raw)
Category: Direct LLM, single-turn (cloud)
"""
import os
import sys
import time
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import make_result, log

PRICE_PER_1M_INPUT = 3.0
PRICE_PER_1M_OUTPUT = 15.0


def extract_code_from_response(text):
    for pattern in [r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return "\n\n".join(matches)
    return None


def run(prompt, task_config, work_dir, output_dir):
    task_id = task_config.get("task_id", "?")

    try:
        import anthropic
    except ImportError:
        return make_result("claude_api_raw", task_id, error="anthropic package not installed. Run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return make_result("claude_api_raw", task_id, error="ANTHROPIC_API_KEY not set")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    log(f"[claude_api_raw] Starting {task_id}")

    data_files = (
        [f for f in os.listdir(work_dir) if f.endswith((".csv", ".parquet"))]
        if os.path.exists(work_dir) else []
    )
    data_context = f"Data files available: {', '.join(data_files)}" if data_files else ""

    full_prompt = (
        f"{prompt}\n\n"
        f"{data_context}\n"
        "Write complete, self-contained Python code that can be run as-is. "
        "Include all imports. Use pandas for data loading. "
        "The data file is in the current directory."
    )

    start = time.perf_counter()
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            temperature=0,
            messages=[{"role": "user", "content": full_prompt}],
        )
        elapsed = time.perf_counter() - start

        text = response.content[0].text
        usage = response.usage
        cost_usd = (
            usage.input_tokens / 1_000_000 * PRICE_PER_1M_INPUT
            + usage.output_tokens / 1_000_000 * PRICE_PER_1M_OUTPUT
        )

        code = extract_code_from_response(text)
        if code:
            with open(os.path.join(output_dir, "solution.py"), "w") as f:
                f.write(code)

        log(f"[claude_api_raw] {task_id} complete | {elapsed:.1f}s | tokens={usage.input_tokens + usage.output_tokens} | cost=${cost_usd:.4f}")

        return make_result(
            agent_id="claude_api_raw",
            task_id=task_id,
            generated_code=code,
            wall_clock_sec=elapsed,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            tokens_used=usage.input_tokens + usage.output_tokens,
            cost_usd=cost_usd,
            raw_output=text,
        )

    except Exception as e:
        elapsed = time.perf_counter() - start
        log(f"[claude_api_raw] {task_id} FAILED after {elapsed:.1f}s: {e}")
        return make_result("claude_api_raw", task_id, wall_clock_sec=elapsed, error=str(e))