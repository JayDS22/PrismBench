"""
Agent: Claude Code
Category: Agentic Coding System (local + cloud)
KEY COMPARISON: Same model as claude_api_raw but with file access, bash, self-correction.
"""
import os
import sys
import time
import json
import subprocess
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import make_result, log


def extract_code_from_output(output_text):
    """Extract Python code from Claude Code output."""
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, output_text, re.DOTALL)
    if matches:
        return "\n\n".join(matches)
    return None


def run(prompt, task_config, work_dir, output_dir):
    """
    Run Claude Code in non-interactive CLI mode.

    Claude Code gets:
    - Full file system access to work_dir
    - Bash execution capability
    - Self-correction (reads errors, fixes code, re-runs)

    This is what makes it fundamentally different from claude_api_raw.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return make_result("claude_code", task_config.get("task_id", "?"),
                           error="ANTHROPIC_API_KEY not set")

    # Check if claude CLI is available
    try:
        version_check = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=10)
        if version_check.returncode != 0:
            raise FileNotFoundError("claude CLI not found")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return make_result("claude_code", task_config.get("task_id", "?"),
                           error="Claude Code CLI not installed. Run: npm install -g @anthropic-ai/claude-code")

    # Build the full prompt
    data_files = [f for f in os.listdir(work_dir) if f.endswith((".csv", ".parquet"))] if os.path.exists(work_dir) else []

    full_prompt = f"""You are working in the directory: {work_dir}
Data files available: {', '.join(data_files)}

{prompt}

Save all output files (code, results, plots) to: {output_dir}
Write your Python code to a file called 'solution.py' in the output directory."""

    start = time.perf_counter()
    try:
        result = subprocess.run(
            [
                "claude",
                "--model", "sonnet",
                "--print", full_prompt,
                "--allowedTools", "Bash,Read,Write,Edit",
                "--output-format", "text",
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=work_dir,
            env={**os.environ, "ANTHROPIC_API_KEY": api_key},
        )
        elapsed = time.perf_counter() - start

        output_text = result.stdout or ""
        error_text = result.stderr or ""

        if result.returncode != 0 and not output_text:
            return make_result("claude_code", task_config.get("task_id", "?"),
                               wall_clock_sec=elapsed, error=f"CLI error: {error_text[:500]}")

        code = extract_code_from_output(output_text)

        # Also check if solution.py was created in output_dir
        solution_path = os.path.join(output_dir, "solution.py")
        if not code and os.path.exists(solution_path):
            with open(solution_path, "r") as f:
                code = f.read()

        # Token counting: Claude Code doesn't expose tokens directly in --print mode
        # Estimate from output length (rough: ~4 chars per token)
        est_output_tokens = len(output_text) // 4
        est_input_tokens = len(full_prompt) // 4

        return make_result(
            agent_id="claude_code",
            task_id=task_config.get("task_id", "?"),
            generated_code=code,
            wall_clock_sec=elapsed,
            tokens_used=est_input_tokens + est_output_tokens,
            input_tokens=est_input_tokens,
            output_tokens=est_output_tokens,
            raw_output=output_text[:10000],  # Truncate to avoid huge files
        )

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return make_result("claude_code", task_config.get("task_id", "?"),
                           wall_clock_sec=elapsed, error="Timeout (>300s)")
    except Exception as e:
        elapsed = time.perf_counter() - start
        return make_result("claude_code", task_config.get("task_id", "?"),
                           wall_clock_sec=elapsed, error=str(e))
