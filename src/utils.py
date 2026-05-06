"""
Timing, logging, file I/O, config loading.
"""
import os
import json
import time
import yaml
import logging
from pathlib import Path
from datetime import datetime
from functools import wraps
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR  = PROJECT_ROOT / "configs"
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED   = PROJECT_ROOT / "data" / "processed"
DATA_ADVERSARIAL = PROJECT_ROOT / "data" / "adversarial"
RESULTS_DIR  = PROJECT_ROOT / "results"
AGENTS_DIR   = PROJECT_ROOT / "agents"

load_dotenv(PROJECT_ROOT / "environment" / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
#a plain callable so agents can call log("msg") rather than log.info("msg")
log = logging.getLogger("prismbench").info


def load_yaml(filename):
    path = CONFIGS_DIR / filename
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_agents_config():
    raw = load_yaml("agents.yaml")
    return {a["id"]: a for a in raw["agents"]}

def load_datasets_config():
    raw = load_yaml("datasets.yaml")
    return {d: info for d, info in raw["datasets"].items()}

def load_tasks_config():
    return load_yaml("tasks.yaml")


class Timer:
    """Context manager for wall-clock timing."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


def timed(func):
    """Decorator that logs execution time of the wrapped function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        log(f"{func.__name__} completed in {time.perf_counter() - start:.2f}s")
        return result
    return wrapper


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log(f"Saved: {path}")

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def get_result_dir(agent_id, task_id, run_id):
    d = RESULTS_DIR / agent_id / task_id / f"run_{run_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def make_result(agent_id, task_id, run_id=1, **kwargs):
    return {
        "agent": agent_id,
        "task_id": task_id,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "generated_code": None,
        "predictions": None,
        "y_true": None,
        "wall_clock_sec": 0.0,
        "cost_usd": 0.0,
        "tokens_used": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "raw_output": None,
        "error": None,
        **kwargs,
    }