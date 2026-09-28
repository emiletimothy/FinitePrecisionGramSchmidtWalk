"""Put the repo root on sys.path so `from core import ...` works when a
script in this directory is run directly (`python scripts/foo.py`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
