"""
Agent: smolagents (HuggingFace)
Category: Code-first iterative agent
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import make_result, log
from src.cost_tracker import calculate_cost

def run(prompt, task_config, work_dir, output_dir):
    try:
        from smolagents import CodeAgent, OpenAIServerModel
    except ImportError:
        return make_result("smolagents", task_config.get("task_id", "?"), error="smolagents not installed")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return make_result("smolagents", task_config.get("task_id", "?"), error="OPENAI_API_KEY not set")

    data_files = [f for f in os.listdir(work_dir) if f.endswith((".csv", ".parquet"))] if os.path.exists(work_dir) else []
    full_prompt = f"{prompt}\n\nData files in {work_dir}: {', '.join(data_files)}"

    start = time.perf_counter()
    try:
        model = OpenAIServerModel(model_id="gpt-4o", api_key=api_key)
        agent = CodeAgent(tools=[], model=model, additional_authorized_imports=["pandas", "numpy", "sklearn", "matplotlib", "seaborn", "shap"])
        result_text = agent.run(full_prompt)
        elapsed = time.perf_counter() - start

        in_tok  = getattr(agent.monitor, "total_input_token_count", 0) or 0
        out_tok = getattr(agent.monitor, "total_output_token_count", 0) or 0
        cost_usd = calculate_cost("smolagents", in_tok, out_tok)

        return make_result(agent_id="smolagents", task_id=task_config.get("task_id", "?"),
                           generated_code=None, wall_clock_sec=elapsed,
                           input_tokens=in_tok, output_tokens=out_tok,
                           tokens_used=in_tok + out_tok, cost_usd=cost_usd,
                           raw_output=str(result_text)[:10000])
    except Exception as e:
        return make_result("smolagents", task_config.get("task_id", "?"),
                           wall_clock_sec=time.perf_counter() - start, error=str(e))
