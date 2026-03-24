"""Safe REPL executor for RLM."""

import functools
import io
import json
import math
import re
from collections import Counter, defaultdict
from typing import Any


class REPLError(Exception):
    """Error during REPL execution."""


class REPLExecutor:
    """Sandboxed Python code executor.

    Provides a restricted execution environment with whitelisted builtins
    and modules. Context variables persist across calls via the shared env dict.
    """

    def __init__(self, max_output_chars: int = 50000):
        self.max_output_chars = max_output_chars

    def execute(self, code: str, env: dict[str, Any]) -> str:
        """Execute Python code in a restricted environment.

        Args:
            code: Python code to execute.
            env: Shared environment dict (context, query, re, etc.).
                 New variables are written back here to persist across calls.

        Returns:
            String output from execution (stdout or last expression value).

        Raises:
            REPLError: If code execution fails.
        """
        code = self._extract_code(code)
        if not code.strip():
            return "No code to execute"

        captured = io.StringIO()

        safe_builtins: dict[str, Any] = {
            # Types
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "frozenset": frozenset,
            "bytes": bytes,
            "bytearray": bytearray,
            # Iteration
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "reversed": reversed,
            "iter": iter,
            "next": next,
            # Aggregation
            "sorted": sorted,
            "sum": sum,
            "min": min,
            "max": max,
            "any": any,
            "all": all,
            # Math
            "abs": abs,
            "round": round,
            "pow": pow,
            "divmod": divmod,
            # String / repr
            "chr": chr,
            "ord": ord,
            "hex": hex,
            "oct": oct,
            "bin": bin,
            "repr": repr,
            "ascii": ascii,
            "format": format,
            # Type checking
            "isinstance": isinstance,
            "issubclass": issubclass,
            "callable": callable,
            "type": type,
            "hasattr": hasattr,
            "getattr": getattr,
            "setattr": setattr,
            # Exceptions
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "StopIteration": StopIteration,
            "Exception": Exception,
            # I/O — use a per-call buffer so parallel executions don't collide
            "print": functools.partial(print, file=captured),
            # Constants
            "True": True,
            "False": False,
            "None": None,
        }

        restricted_globals: dict[str, Any] = {"__builtins__": safe_builtins}
        restricted_globals.update(
            {
                "re": re,
                "json": json,
                "math": math,
                "Counter": Counter,
                "defaultdict": defaultdict,
            }
        )
        restricted_globals.update(env)

        local_env: dict[str, Any] = {}

        try:
            exec(
                compile(code, "<repl>", "exec"), restricted_globals, local_env
            )  # noqa: S102
            output = captured.getvalue()

            # Try to evaluate the last line as an expression
            lines = code.strip().split("\n")
            if lines:
                last = lines[-1].strip()
                skip_keywords = [
                    "=",
                    "import",
                    "def",
                    "class",
                    "if",
                    "for",
                    "while",
                    "with",
                    "print(",
                    "return",
                    "raise",
                ]
                if last and not any(kw in last for kw in skip_keywords):
                    try:
                        merged = {**restricted_globals, **local_env}
                        result = eval(last, merged)  # noqa: S307
                        if result is not None:
                            output += str(result) + "\n"
                    except Exception:
                        pass

            # Persist new variables back to the shared env
            env.update(local_env)

            if not output:
                return "Code executed successfully (no output)"

            if len(output) > self.max_output_chars:
                truncated = output[: self.max_output_chars]
                return (
                    f"{truncated}\n\n"
                    f"[Output truncated: {len(output)} chars total, "
                    f"showing first {self.max_output_chars}]"
                )

            return output.strip()

        except Exception as e:
            raise REPLError(f"Execution error: {e}") from e

    def _extract_code(self, text: str) -> str:
        """Extract code from markdown code blocks if present."""
        if "```python" in text:
            start = text.find("```python") + len("```python")
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()

        return text
