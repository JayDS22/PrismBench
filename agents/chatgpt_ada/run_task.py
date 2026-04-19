"""
Agent: ChatGPT Advanced Data Analysis (GPT-4o)
Category: LLM Notebook Agent (cloud)
"""
import os, sys, time, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import make_result, log


def extract_code(text):
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return "\n\n".join(matches) if matches else None


def run(prompt, task_config, work_dir, output_dir):
    import openai
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return make_result("chatgpt_ada", task_config.get("task_id", "?"), error="OPENAI_API_KEY not set")

    client = openai.OpenAI(api_key=api_key)
    data_files = [f for f in os.listdir(work_dir) if f.endswith((".csv", ".parquet"))] if os.path.exists(work_dir) else []
    full_prompt = f"""{prompt}\n\nData files available: {', '.join(data_files)}\nWrite complete, self-contained Python code with all imports."""

    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0,
            max_tokens=4096,
        )
        elapsed = time.perf_counter() - start
        text = response.choices[0].message.content
        usage = response.usage
        code = extract_code(text)
        if code:
            with open(os.path.join(output_dir, "solution.py"), "w") as f:
                f.write(code)
        return make_result(
            agent_id="chatgpt_ada", task_id=task_config.get("task_id", "?"),
            generated_code=code, wall_clock_sec=elapsed,
            tokens_used=usage.total_tokens, input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens, raw_output=text,
        )
    except Exception as e:
        return make_result("chatgpt_ada", task_config.get("task_id", "?"),
                           wall_clock_sec=time.perf_counter() - start, error=str(e))
