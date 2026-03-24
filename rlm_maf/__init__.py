"""Recursive Language Models (RLM) - Microsoft Agent Framework port.

Based on the RLM paper by Alex Zhang and Omar Khattab (MIT, 2025).
Original implementation: https://github.com/ysz/recursive-llm
"""

from .core import RLM, RLMStats
from .repl import REPLError, REPLExecutor

__all__ = ["RLM", "RLMStats", "REPLError", "REPLExecutor"]
