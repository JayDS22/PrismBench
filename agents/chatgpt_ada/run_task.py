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


def _localize_container_paths(code):
    """Rewrite OpenAI code-interpreter container paths to bare filenames.

    Two patterns occur in ADA-generated code:
      - Inputs:  /mnt/data/file-<id>-<name>.<ext>  (uploaded files)
      - Outputs: /mnt/data/<name>.<ext>            (files the model writes)

    Both don't exist when task_runner post-executes solution.py locally.
    Stripping the directory prefix (and any file-<id>- token) leaves a bare
    filename relative to work_dir, which is the agent's CWD during post-exec.
    """
    if not code:
        return code
    return re.sub(
        r"/mnt/data/(?:file-[A-Za-z0-9]+-)?([^'\"\s]+)",
        r"\1",
        code,
    )


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
            file_hint = (
                f"\n\nData files are mounted in the code-interpreter "
                f"container at /mnt/data/: {names}.\n\n"
                "IMPORTANT: Complete the entire task in this single response — "
                "load data, train models, evaluate, save all required output "
                "files (predictions.csv, metrics.json, etc.), and finally "
                "return the full self-contained solution as a ```python``` "
                "block. Do not stop after exploration; continue executing "
                "until every output the task asks for has been produced."
            )

        log(f"[chatgpt_ada] {task_id} | files={len(uploaded_file_ids)}")

        instructions = (
            "You are a data science assistant with access to a Python code "
            "interpreter and the user's data files. Write complete Python "
            "code that solves the task end-to-end (explore, train, evaluate, "
            "save outputs), execute each step in the interpreter to verify "
            "it works, and produce all artifacts the task asks for "
            "(predictions.csv, metrics.json, etc.). Put the final "
            "self-contained solution in a ```python``` code block as the "
            "LAST thing in your reply."
        )
        tool_spec = [{"type": "code_interpreter", "container": container}]

        resp = client.responses.create(
            model="gpt-4o",
            tools=tool_spec,
            instructions=instructions,
            input=prompt + file_hint,
            max_output_tokens=8192,
            max_tool_calls=20,
        )

        # The Responses API loop with code_interpreter often bails after the
        # first tool execution on multi-step tasks. If the agent didn't
        # generate the file outputs the prompt asked for, chain a follow-up
        # response with previous_response_id to push it to continue. Up to
        # 2 continuations (3 calls total) before giving up.
        all_executed = [_extract_executed_code(resp) or ""]
        all_text     = [resp.output_text or ""]
        in_tok_total = getattr(getattr(resp, "usage", None), "input_tokens",  0) or 0
        out_tok_total = getattr(getattr(resp, "usage", None), "output_tokens", 0) or 0

        for attempt in range(2):
            joined_code = "\n".join(all_executed).lower()
            if "predictions.csv" in joined_code or "to_csv" in joined_code:
                break  # task likely complete
            log(f"[chatgpt_ada] continuation {attempt + 1}: pushing model to finish")
            resp = client.responses.create(
                model="gpt-4o",
                previous_response_id=resp.id,
                tools=tool_spec,
                input=(
                    "Continue executing now. Train at least three classifiers "
                    "on the data, evaluate each on the test set, pick the best, "
                    "and save predictions.csv (columns: y_true, y_pred, y_prob) "
                    "and metrics.json to /mnt/data/. Do not stop until "
                    "predictions.csv has been written."
                ),
                max_output_tokens=8192,
                max_tool_calls=15,
            )
            all_executed.append(_extract_executed_code(resp) or "")
            all_text.append(resp.output_text or "")
            in_tok_total  += getattr(getattr(resp, "usage", None), "input_tokens",  0) or 0
            out_tok_total += getattr(getattr(resp, "usage", None), "output_tokens", 0) or 0

        elapsed = time.perf_counter() - start
        full_text = "\n\n".join(all_text)
        # Prefer the final markdown block; fall back to concatenated executed code.
        code = _extract_python_blocks(full_text) or "\n\n".join(c for c in all_executed if c)
        code = _localize_container_paths(code)

        if code:
            with open(os.path.join(output_dir, "solution.py"), "w") as f:
                f.write(code)

        # Use accumulated counts across all chained responses (initial +
        # any continuations), not just the last one.
        in_tok  = in_tok_total
        out_tok = out_tok_total
        tot_tok = in_tok + out_tok

        log(f"[chatgpt_ada] done {elapsed:.1f}s | tokens={tot_tok} | code={'Y' if code else 'N'} | calls={len(all_text)}")

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
