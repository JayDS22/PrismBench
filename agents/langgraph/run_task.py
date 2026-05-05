"""
Agent: LangGraph
Category: Stateful Agent Graphs
"""
import os, sys, time, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import make_result, log


def extract_code(text):
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return "\n\n".join(matches) if matches else None


def run(prompt, task_config, work_dir, output_dir):
    try:
        from langgraph.prebuilt import create_react_agent
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        return make_result("langgraph", task_config.get("task_id", "?"), error=str(e))

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return make_result("langgraph", task_config.get("task_id", "?"), error="ANTHROPIC_API_KEY not set")

    data_files = [f for f in os.listdir(work_dir) if f.endswith((".csv", ".parquet"))] if os.path.exists(work_dir) else []
    full_prompt = f"""{prompt}\n\nData files available: {', '.join(data_files)}\nWrite complete, self-contained Python code with all imports."""

    start = time.perf_counter()
    try:
        llm = ChatAnthropic(model="claude-sonnet-4-6", api_key=api_key, temperature=0, max_tokens=4096)
        agent = create_react_agent(llm, tools=[])
        result = agent.invoke({"messages": [{"role": "user", "content": full_prompt}]})
        elapsed = time.perf_counter() - start

        last_message = result["messages"][-1]
        text = last_message.content if hasattr(last_message, "content") else str(last_message)

        usage = last_message.usage_metadata if hasattr(last_message, "usage_metadata") else {}
        input_tokens = usage.get("input_tokens", 0) if isinstance(usage, dict) else 0
        output_tokens = usage.get("output_tokens", 0) if isinstance(usage, dict) else 0

        code = extract_code(text)
        if code:
            with open(os.path.join(output_dir, "solution.py"), "w") as f:
                f.write(code)

        return make_result(
            agent_id="langgraph", task_id=task_config.get("task_id", "?"),
            generated_code=code, wall_clock_sec=elapsed,
            tokens_used=input_tokens + output_tokens,
            input_tokens=input_tokens, output_tokens=output_tokens,
            raw_output=text[:10000],
        )
    except Exception as e:
        return make_result("langgraph", task_config.get("task_id", "?"),
                           wall_clock_sec=time.perf_counter() - start, error=str(e))
