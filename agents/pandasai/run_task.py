"""
Category: Lightweight NL->pandas (local)
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import make_result, log
from src.cost_tracker import calculate_cost
import pandas as pd


def _make_usage_tracking_llm(api_key):
    """Subclass pandasai_openai.OpenAI to capture cumulative token usage
    across every chat_completion call SmartDataframe.chat() makes."""
    from pandasai_openai import OpenAI

    class _UsageTrackingOpenAI(OpenAI):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cumulative_usage = {"input_tokens": 0, "output_tokens": 0}

        def chat_completion(self, value, memory):
            messages = memory.to_openai_messages() if memory else []
            messages.append({"role": "user", "content": value})
            params = {**self._invocation_params, "messages": messages}
            if self.stop is not None:
                params["stop"] = [self.stop]
            response = self.client.create(**params)
            usage = getattr(response, "usage", None)
            if usage:
                self.cumulative_usage["input_tokens"]  += getattr(usage, "prompt_tokens",     0) or 0
                self.cumulative_usage["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
            return response.choices[0].message.content

    return _UsageTrackingOpenAI(api_token=api_key)


def run(prompt, task_config, work_dir, output_dir):
    try:
        from pandasai import SmartDataframe
        from pandasai_openai import OpenAI  # noqa: F401  (validate install)
    except ImportError as e:
        return make_result("pandasai", task_config.get("task_id", "?"),
                           error=f"pandasai/pandasai-openai not installed: {e}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return make_result("pandasai", task_config.get("task_id", "?"), error="OPENAI_API_KEY not set")

    data_files = [f for f in os.listdir(work_dir) if f.endswith((".csv", ".parquet"))] if os.path.exists(work_dir) else []
    if not data_files:
        return make_result("pandasai", task_config.get("task_id", "?"), error="No data files")

    data_file = os.path.join(work_dir, data_files[0])
    df = pd.read_csv(data_file) if data_file.endswith(".csv") else pd.read_parquet(data_file)

    # Build the LLM up front so cumulative_usage is captured even if
    # SmartDataframe.chat() raises (it retries internally and burns tokens
    # before raising, so the partial counts matter for D5).
    llm = _make_usage_tracking_llm(api_key)
    sdf = SmartDataframe(df, config={"llm": llm})

    start = time.perf_counter()
    result_text = None
    err = None
    try:
        result_text = sdf.chat(prompt)
    except Exception as e:
        err = str(e)
    elapsed = time.perf_counter() - start

    in_tok  = llm.cumulative_usage["input_tokens"]
    out_tok = llm.cumulative_usage["output_tokens"]
    cost_usd = calculate_cost("pandasai", in_tok, out_tok)

    return make_result(
        agent_id="pandasai", task_id=task_config.get("task_id", "?"),
        generated_code=None, wall_clock_sec=elapsed,
        input_tokens=in_tok, output_tokens=out_tok,
        tokens_used=in_tok + out_tok, cost_usd=cost_usd,
        raw_output=str(result_text or "")[:10000],
        error=err,
    )
