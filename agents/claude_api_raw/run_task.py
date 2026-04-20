"""
Agent: Claude API (raw)
Category: Direct LLM, single-turn (cloud)
BASELINE for Claude Code comparison — same model, no agentic scaffolding.
"""
import os
import sys
import time
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import make_result, log


def extract_code_from_response(text):
    """Extract Python code blocks from LLM response."""
    # Find all ```python ... ``` blocks
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(matches)
    # Fallback: find any ``` ... ``` blocks
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(matches)
    return None


def run(prompt, task_config, work_dir, output_dir):
    """
    Run Claude API (raw single-turn) on a task.
    Sends the prompt, gets back text (hopefully with code), extracts code.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return make_result("claude_api_raw", task_config.get("task_id", "?"),
                           error="ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    # Build the full prompt with data context
    data_files = [f for f in os.listdir(work_dir) if f.endswith((".csv", ".parquet"))] if os.path.exists(work_dir) else []
    data_context = f"Data files available: {', '.join(data_files)}" if data_files else ""

    full_prompt = f"""{prompt}

{data_context}

IMPORTANT: Write complete, self-contained Python code that can be run as-is.
Include all imports. Use pandas for data loading.
The data file is in the current directory."""

    start = time.perf_counter()
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            temperature=0,
            messages=[{"role": "user", "content": full_prompt}],
        )
        text = response.content[0].text
        usage = response.usage
        elapsed = time.perf_counter() - start

        code = extract_code_from_response(text)

        # Save the generated code
        if code:
            code_path = os.path.join(output_dir, "solution.py")
            with open(code_path, "w") as f:
                f.write(code)

        return make_result(
            agent_id="claude_api_raw",
            task_id=task_config.get("task_id", "?"),
            generated_code=code,
            wall_clock_sec=elapsed,
            tokens_used=usage.input_tokens + usage.output_tokens,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            raw_output=text,
        )

    except Exception as e:
        elapsed = time.perf_counter() - start
        return make_result("claude_api_raw", task_config.get("task_id", "?"),
                           wall_clock_sec=elapsed, error=str(e))
