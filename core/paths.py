"""Output locations: figures/<experiment>/<name>.png, caches beside them."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


def figure_path(experiment: str, filename: str) -> str:
    """Default save path for a figure (and its derived cache), grouped by
    experiment name under figures/."""
    d = FIGURES / experiment
    d.mkdir(parents=True, exist_ok=True)
    return str(d / filename)
