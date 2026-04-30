"""
DS-RouteBench Agent: LangGraph
Category: Stateful Agent Graphs (LOW PRIORITY)
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import make_result

def run(prompt, task_config, work_dir, output_dir):
    # LangGraph requires more complex setup — implement in Phase 2
    return make_result("langgraph", task_config.get("task_id", "?"),
                       error="LangGraph wrapper not yet implemented. Lower priority agent.")
