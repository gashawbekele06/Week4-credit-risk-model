# src/config.py
from pathlib import Path

# Project root (works regardless of where script is run)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data path - adjust if your file is in data/raw or elsewhere
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"

# Alternative if you have raw folder
# DATA_PATH = PROJECT_ROOT / "data" / "raw" / "training.csv"

# Plot style
PLOT_STYLE = "seaborn-v0_8"
PALETTE = "husl"