"""Directories of the paper material, independent of the current working directory."""

from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
DATA_DIR = PAPER_DIR / "data"
FIGURES_DIR = PAPER_DIR / "figures"
