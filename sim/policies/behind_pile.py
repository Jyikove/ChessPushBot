"""Compatibility entry point for the MuJoCo strategy worker."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from duelbot.planning.policy import Policy

__all__ = ["Policy"]
