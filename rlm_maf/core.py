"""Core RLM implementation using Microsoft Agent Framework."""

import asyncio
import concurrent.futures
import re
from dataclasses import dataclass, field
from typing import Annotated, Any

from agent_framework import FunctionInvocationContext, tool
from pydantic import Field

from .prompts import build_system_prompt
from .repl import REPLError, REPLExecutor


@dataclass
class RLMStats:
    """Accumulated token usage and call statistics across recursive calls."""

    llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    max_depth_reached: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


@tool(
    name="execute_python",
    description=(
        "Execute Python code in the REPL environment. "
        "Variables available: context (str), query (str), re (regex module), "
        "recursive_llm(sub_query, sub_context) -> str. "
        "Use print() to see output."
    ),
    approval_mode="never_require",
)
def execute_python(
    code: Annotated[str, Field(description="Python code to execute in the REPL")],
    ctx: FunctionInvocationContext,
) -> str:
    """Run code in the sandboxed REPL and return output."""
    repl: REPLExecutor = ctx.kwargs["repl"]
    env: dict = ctx.kwargs["env"]
    try:
        return repl.execute(code, env)
    except REPLError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


class RLM:
    """Recursive Language Model using Microsoft Agent Framework.

    Processes unbounded-length context by storing it as a variable
    and letting the LLM explore it via a Python REPL tool.
    Supports recursive sub-calls at increasing depth.
    """

    def __init__(
        self,
        client: Any,
        max_depth: int = 5,
    ):
        self.client = client
        self.max_depth = max_depth
        self._stats = RLMStats()

    @property
    def stats(self) -> RLMStats:
        """Token usage and call statistics from the last complete() call."""
        return self._stats

    async def complete(self, query: str, context: str) -> str:
        """Process a query against a long context.

        Args:
            query: The question to answer.
            context: The document text (stored as variable, not in prompt).

        Returns:
            The final answer string.
        """
        self._stats = RLMStats()
        return await self._complete_internal(query, context, depth=0)

    async def _complete_internal(self, query: str, context: str, depth: int) -> str:
        if depth >= self.max_depth:
            return f"Max recursion depth ({self.max_depth}) reached"

        repl = REPLExecutor()
        env: dict[str, Any] = {
            "context": context,
            "query": query,
            "re": re,
            "recursive_llm": self._make_recursive_fn(depth),
        }

        agent = self.client.as_agent(
            name=f"RLM_depth{depth}",
            instructions=build_system_prompt(len(context), depth),
            tools=[execute_python],
        )

        result = await agent.run(
            query,
            function_invocation_kwargs={
                "repl": repl,
                "env": env,
            },
        )

        # Accumulate token usage stats
        async with self._stats._lock:
            self._stats.llm_calls += 1
            self._stats.max_depth_reached = max(self._stats.max_depth_reached, depth)
            if result.usage_details:
                self._stats.total_input_tokens += (
                    result.usage_details.get("input_token_count") or 0
                )
                self._stats.total_output_tokens += (
                    result.usage_details.get("output_token_count") or 0
                )

        return result.text

    def _make_recursive_fn(self, depth: int):
        """Create a sync recursive_llm function for the REPL environment.

        This function is called from inside exec() during a tool invocation,
        which is already running within an async event loop. Since we need to
        call async code (_complete_internal) from this sync context, we spawn
        a new thread with its own event loop via asyncio.run().
        """
        rlm = self

        def recursive_llm(sub_query: str, sub_context: str) -> str:
            """Recursively process a sub-section of context."""
            if depth + 1 >= rlm.max_depth:
                return f"Max recursion depth ({rlm.max_depth}) reached"

            # We're always inside an async event loop here (MAF agent loop),
            # so we must run the nested async call in a separate thread.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    asyncio.run,
                    rlm._complete_internal(sub_query, sub_context, depth + 1),
                )
                return future.result()

        return recursive_llm
