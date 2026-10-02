"""Standalone EPLB control-plane overhead plugin for vLLM-HUST + vLLM-Ascend-HUST."""

from ._version import __version__
from .plugin import register

__all__ = ["__version__", "register"]
