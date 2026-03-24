"""Basic usage example for RLM (MAF port).

Demonstrates how to use the Recursive Language Model to answer complex
analytical queries against a long document. The document is stored as a
variable and explored programmatically via a sandboxed Python REPL,
rather than being passed directly in the prompt.

Usage:
    uv run basic_usage.py
"""

import asyncio
import os
import time
from pathlib import Path

from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

from rlm_maf import RLM

load_dotenv()

# Load document from file
long_document = (Path(__file__).parent / "data" / "ai_history.txt").read_text()


async def main() -> None:
    """Run basic RLM example."""
    client = AzureOpenAIResponsesClient(
        project_endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
        deployment_name=os.environ["AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME"],
        credential=AzureCliCredential(),
    )

    rlm = RLM(client=client)

    query = (
        "Analyze this document computationally and report: "
        "(1) The exact number of distinct years mentioned (CE and BCE separately). "
        "(2) The 5 years that appear in the most separate sections/appendices, with the count for each. "
        "(3) Every year where BOTH a hardware/computing milestone AND a software/AI-system milestone occurred. "
        "(4) The longest consecutive gap (in years) where no year is mentioned between 1900 and 2026. "
        "(5) The total number of named AI systems or projects mentioned across the entire document."
    )

    print(f"Query: {query}")
    print(f"Context length: {len(long_document):,} characters")
    print("\nProcessing with RLM...\n")

    start = time.perf_counter()
    result = await rlm.complete(query, long_document)
    elapsed = time.perf_counter() - start

    print("Result:")
    print(result)

    s = rlm.stats
    print(f"\nToken Usage:")
    print(f"  LLM calls:     {s.llm_calls}")
    print(f"  Input tokens:  {s.total_input_tokens:,}")
    print(f"  Output tokens: {s.total_output_tokens:,}")
    print(f"  Max depth:     {s.max_depth_reached}")
    print(f"\nTime: {elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
