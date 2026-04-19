"""
DS-RouteBench: Cost Tracker
Calculate API costs per agent per task based on token usage.
"""
from src.utils import load_agents_config

# Pricing loaded from agents.yaml, but we keep a fast lookup here
_AGENT_PRICING_CACHE = None

def _get_pricing():
    global _AGENT_PRICING_CACHE
    if _AGENT_PRICING_CACHE is None:
        agents = load_agents_config()
        _AGENT_PRICING_CACHE = {
            aid: {
                "input": a.get("pricing_input_per_M", 0),
                "output": a.get("pricing_output_per_M", 0),
            }
            for aid, a in agents.items()
        }
    return _AGENT_PRICING_CACHE


def calculate_cost(agent_id, input_tokens=0, output_tokens=0):
    """
    Calculate USD cost for an API call.
    Returns float (dollars).
    """
    pricing = _get_pricing()
    p = pricing.get(agent_id, {"input": 0, "output": 0})
    cost = (input_tokens * p["input"] / 1_000_000) + (output_tokens * p["output"] / 1_000_000)
    return round(cost, 6)


def format_cost(cost_usd):
    """Pretty-print a cost value."""
    if cost_usd == 0:
        return "free (local)"
    elif cost_usd < 0.01:
        return f"${cost_usd:.4f}"
    else:
        return f"${cost_usd:.2f}"
