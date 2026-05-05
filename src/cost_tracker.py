"""
Logic: Cost Tracker
Calculate API costs (D5 USD) and CO2 emissions (D5 carbon) per agent per task.

Carbon is measured locally with CodeCarbon for agents marked `local_or_cloud: local`,
estimated from token count for cloud agents, and summed for hybrid (`both`) agents.
The cloud estimate uses 0.5 kg CO2 per million tokens, in the range reported by
Luccioni et al. (2023) and Patterson et al. (2021) for inference on GPT-4 /
Claude Sonnet-class models.
"""
from src.utils import load_agents_config

_AGENT_PRICING_CACHE = None
_AGENT_LOCATION_CACHE = None

CLOUD_KG_CO2_PER_M_TOKENS = 0.5


def _get_pricing():
    global _AGENT_PRICING_CACHE
    if _AGENT_PRICING_CACHE is None:
        agents = load_agents_config()
        _AGENT_PRICING_CACHE = {
            aid: {
                "input":  a.get("pricing_input_per_M", 0),
                "output": a.get("pricing_output_per_M", 0),
            }
            for aid, a in agents.items()
        }
    return _AGENT_PRICING_CACHE


def _get_location(agent_id):
    global _AGENT_LOCATION_CACHE
    if _AGENT_LOCATION_CACHE is None:
        agents = load_agents_config()
        _AGENT_LOCATION_CACHE = {
            aid: a.get("local_or_cloud", "cloud") for aid, a in agents.items()
        }
    return _AGENT_LOCATION_CACHE.get(agent_id, "cloud")


def calculate_cost(agent_id, input_tokens=0, output_tokens=0):
    """USD cost for one agent run."""
    p = _get_pricing().get(agent_id, {"input": 0, "output": 0})
    cost = (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000
    return round(cost, 6)


def calculate_carbon_kg(agent_id, input_tokens=0, output_tokens=0, measured_kg=None):
    """kg CO2 for one agent run.

    Returns (carbon_kg, source). `source` is "measured" for local agents,
    "estimated" for cloud agents, "hybrid" when both apply.
    """
    location = _get_location(agent_id)
    total_tokens = (input_tokens or 0) + (output_tokens or 0)
    cloud_estimate_kg = total_tokens / 1_000_000 * CLOUD_KG_CO2_PER_M_TOKENS

    if location == "local":
        return round(measured_kg or 0.0, 9), "measured"
    if location == "cloud":
        return round(cloud_estimate_kg, 9), "estimated"
    return round((measured_kg or 0.0) + cloud_estimate_kg, 9), "hybrid"


def format_cost(cost_usd):
    """Pretty-print a cost value."""
    if cost_usd == 0:
        return "free (local)"
    if cost_usd < 0.01:
        return f"${cost_usd:.4f}"
    return f"${cost_usd:.2f}"
