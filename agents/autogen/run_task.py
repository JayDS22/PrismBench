"""
Microsoft AutoGen — Multi-Agent Framework (pyautogen v0.2 API)

NOTE: AutoGen stable (v0.4) uses a fully async API under `autogen-agentchat`
+ `autogen-ext`. Migrate by rewriting `run()` as `async def` and using
AssistantAgent from `autogen_agentchat.agents` with OpenAIChatCompletionClient.
"""
import os, sys, time, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import make_result, log
from src.cost_tracker import calculate_cost

_SYSTEM_MESSAGE = (
    "You are an expert data science assistant. "
    "Write clean, well-commented Python code using pandas, scikit-learn, "
    "and SHAP where appropriate. "
    "Always save your final, complete solution as a single file. "
    "In your final code block, use '# filename: solution.py' as the very "
    "first line so AutoGen saves it as solution.py in the working directory."
)


def _read_usage_from_client(client) -> tuple[int, int]:
    """Pull aggregated token usage off an autogen OpenAIWrapper.

    `total_usage_summary` is the documented surface; its shape is
    {'total_cost': float, '<model>': {'cost': ..., 'prompt_tokens': ...,
    'completion_tokens': ..., 'total_tokens': ...}, ...}.
    """
    summary = getattr(client, "total_usage_summary", None) or {}
    prompt = completion = 0
    for k, v in summary.items():
        if isinstance(v, dict):
            prompt += v.get("prompt_tokens", 0) or 0
            completion += v.get("completion_tokens", 0) or 0
    return prompt, completion


def _find_solution(output_dir: str, work_dir: str) -> str | None:
    """Search for solution.py in priority order, falling back to the newest
    .py file in AutoGen's default code-execution subdir (work_dir/coding/)."""
    candidates = [
        os.path.join(output_dir, "solution.py"),
        os.path.join(work_dir, "solution.py"),
        os.path.join(work_dir, "coding", "solution.py"),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p) as f:
                return f.read()

    coding_dir = os.path.join(work_dir, "coding")
    if os.path.isdir(coding_dir):
        py_files = sorted(glob.glob(os.path.join(coding_dir, "*.py")), key=os.path.getmtime)
        if py_files:
            with open(py_files[-1]) as f:
                return f.read()

    return None


def run(prompt, task_config, work_dir, output_dir):
    task_id = task_config.get("task_id", "?")

    try:
        import autogen
    except ImportError:
        return make_result("autogen", task_id, error="pyautogen not installed. Run: pip install pyautogen")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return make_result("autogen", task_id, error="OPENAI_API_KEY not set")

    model = task_config.get("model", "gpt-4o")
    max_replies = task_config.get("max_auto_reply", 10)

    log(f"[autogen] Starting {task_id} | model={model} | max_replies={max_replies}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    assistant = autogen.AssistantAgent(
        "ds_assistant",
        llm_config={"config_list": [{"model": model, "api_key": api_key}]},
        system_message=_SYSTEM_MESSAGE,
    )
    user_proxy = autogen.UserProxyAgent(
        "user_proxy",
        human_input_mode="NEVER",
        code_execution_config={
            "work_dir": work_dir,
            "use_docker": False,  # already inside Docker; nested Docker would fail
        },
        max_consecutive_auto_reply=max_replies,
    )

    data_files = (
        [f for f in os.listdir(work_dir) if f.endswith((".csv", ".parquet"))]
        if os.path.exists(work_dir) else []
    )
    full_prompt = (
        f"{prompt}\n\n"
        f"Data files available in {work_dir}: {', '.join(data_files) if data_files else 'none'}.\n"
        f"Use '# filename: solution.py' at the top of your final code block. "
        f"As a last step, copy solution.py to {output_dir}/solution.py."
    )

    start = time.perf_counter()
    try:
        user_proxy.initiate_chat(assistant, message=full_prompt)
        elapsed = time.perf_counter() - start

        # Agent objects are not JSON-serializable; use agent.name as key
        serializable = {
            agent.name: messages
            for agent, messages in user_proxy.chat_messages.items()
        }
        prompt_tokens, completion_tokens = _read_usage_from_client(assistant.client)
        cost_usd = calculate_cost("autogen", prompt_tokens, completion_tokens)
        raw_output = json.dumps(serializable, default=str)

        code = _find_solution(output_dir, work_dir)
        if code is None:
            log(f"[autogen] WARNING: solution.py not found for {task_id}")
        elif not os.path.exists(os.path.join(output_dir, "solution.py")):
            with open(os.path.join(output_dir, "solution.py"), "w") as f:
                f.write(code)

        log(f"[autogen] {task_id} complete | {elapsed:.1f}s | tokens={prompt_tokens + completion_tokens} | cost=${cost_usd:.4f}")

        return make_result(
            agent_id="autogen",
            task_id=task_id,
            generated_code=code,
            wall_clock_sec=elapsed,
            raw_output=raw_output[-10000:],  # tail is more relevant than head
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            tokens_used=prompt_tokens + completion_tokens,
            cost_usd=cost_usd,
        )

    except Exception as e:
        elapsed = time.perf_counter() - start
        log(f"[autogen] {task_id} FAILED after {elapsed:.1f}s: {e}")
        return make_result("autogen", task_id, wall_clock_sec=elapsed, error=str(e))