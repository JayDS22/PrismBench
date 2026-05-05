"""
DS-RouteBench Agent: ChatGPT ADA (Advanced Data Analysis)

Real ADA via the OpenAI Responses API with the `code_interpreter` tool —
the model gets a sandboxed Python container with the user's data files
mounted, executes code internally, and returns the final solution.

OpenAI-side peer to `claude_code` for RQ4 (does agentic scaffolding
generalize across model providers?).
"""
import os, sys, time, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import make_result, log


def _extract_python_blocks(text):
    if not text:
        return None
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    return "\n\n".join(blocks) if blocks else None


def _extract_executed_code(resp):
    """Pull the code actually executed in the code_interpreter container.
    Preferred over markdown-block parsing because ADA usually executes
    rather than restating code, so the markdown block may be absent.
    """
    chunks = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) == "code_interpreter_call":
            code = getattr(item, "code", None)
            if code:
                chunks.append(code)
    return "\n\n".join(chunks) if chunks else None


def run(prompt, task_config, work_dir, output_dir):
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return make_result("chatgpt_ada", task_config.get("task_id", "?"),
                           error="OPENAI_API_KEY not set")

    client = openai.OpenAI(api_key=api_key)
    task_id = task_config.get("task_id", "?")

    data_files = []
    if os.path.exists(work_dir):
        for f in os.listdir(work_dir):
            if f.endswith((".csv", ".parquet", ".json", ".txt")):
                data_files.append(os.path.join(work_dir, f))

    start = time.perf_counter()
    uploaded_file_ids = []

    try:
        for fp in data_files:
            with open(fp, "rb") as fh:
                f_obj = client.files.create(file=fh, purpose="user_data")
                uploaded_file_ids.append(f_obj.id)

        container = {"type": "auto"}
        if uploaded_file_ids:
            container["file_ids"] = uploaded_file_ids

        file_hint = ""
        if data_files:
            names = ", ".join(os.path.basename(p) for p in data_files)
            file_hint = (f"\n\nData files are mounted in the code-interpreter "
                         f"container at /mnt/data/: {names}")

        log(f"[chatgpt_ada] {task_id} | files={len(uploaded_file_ids)}")

        resp = client.responses.create(
            model="gpt-4o",
            tools=[{"type": "code_interpreter", "container": container}],
            instructions=(
                "You are a data science assistant with access to a Python code "
                "interpreter and the user's data files. Write complete Python "
                "code that solves the task and execute it in the interpreter "
                "to verify it works. Put the final solution in a ```python``` "
                "code block as the LAST thing in your reply."
            ),
            input=prompt + file_hint,
        )

        elapsed = time.perf_counter() - start
        full_text = resp.output_text or ""
        code = _extract_executed_code(resp) or _extract_python_blocks(full_text)

        if code:
            with open(os.path.join(output_dir, "solution.py"), "w") as f:
                f.write(code)

        usage = getattr(resp, "usage", None)
        in_tok  = getattr(usage, "input_tokens", 0) if usage else 0
        out_tok = getattr(usage, "output_tokens", 0) if usage else 0
        tot_tok = getattr(usage, "total_tokens", 0) if usage else (in_tok + out_tok)

        log(f"[chatgpt_ada] done {elapsed:.1f}s | tokens={tot_tok} | code={'Y' if code else 'N'}")

        return make_result(
            agent_id="chatgpt_ada", task_id=task_id,
            generated_code=code, wall_clock_sec=elapsed,
            tokens_used=tot_tok, input_tokens=in_tok, output_tokens=out_tok,
            raw_output=full_text[:10000],
        )

    except Exception as e:
        return make_result("chatgpt_ada", task_id,
                           wall_clock_sec=time.perf_counter() - start, error=str(e))
    finally:
        for fid in uploaded_file_ids:
            try:
                client.files.delete(fid)
            except Exception:
                pass
