"""Shared pytest fixtures and path setup."""
import sys
from pathlib import Path

# Make the project root importable so `from src import router` and
# `import prismbench_utils` resolve when pytest is invoked from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
