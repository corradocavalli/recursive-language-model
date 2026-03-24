"""Basic usage example WITHOUT RLM, passes entire document in the prompt.

Serves as a baseline comparison for the RLM approach. The full document
is embedded directly in the LLM prompt, so the model must attend to the
entire context at once. Compare token usage and accuracy against
basic_usage.py to see the benefits of RLM.

Usage:
    uv run basic_usage_no_rlm.py
"""

import asyncio
import os
import time
from pathlib import Path

from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()

# Load document from file (same as basic_usage.py)
long_document = (Path(__file__).parent / "data" / "ai_history.txt").read_text()


async def main() -> None:
    """Send the full document + query directly to the LLM (no RLM)."""
    client = AzureOpenAIResponsesClient(
        project_endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
        deployment_name=os.environ["AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME"],
        credential=AzureCliCredential(),
    )

    query = (
        "Analyze this document computationally and report: "
        "(1) The exact number of distinct years mentioned (CE and BCE separately). "
        "(2) The 5 years that appear in the most separate sections/appendices, with the count for each. "
        "(3) Every year where BOTH a hardware/computing milestone AND a software/AI-system milestone occurred. "
        "(4) The longest consecutive gap (in years) where no year is mentioned between 1900 and 2026. "
        "(5) The total number of named AI systems or projects mentioned across the entire document."
    )

    prompt = f"""Given the following document, answer the question directly with specific numbers and lists. Do NOT ask for permission or outline steps. Provide the final answers immediately.

Document:
{long_document}

Question: {query}"""

    print(f"Query: {query}")
    print(f"Context length: {len(long_document):,} characters")
    print("\nProcessing WITHOUT RLM (full context in prompt)...\n")

    agent = client.as_agent(
        name="direct_llm",
        instructions="You are a precise document analyst. Answer questions directly with specific numbers, lists, and data. Never ask for confirmation or outline plans. Provide complete final answers immediately.",
    )

    start = time.perf_counter()
    result = await agent.run(prompt)
    elapsed = time.perf_counter() - start

    print("Result:")
    print(result.text)

    usage = result.usage_details
    input_tokens = (usage.get("input_token_count") or 0) if usage else 0
    output_tokens = (usage.get("output_token_count") or 0) if usage else 0

    print(f"\nToken Usage:")
    print(f"  LLM calls:     1")
    print(f"  Input tokens:  {input_tokens:,}")
    print(f"  Output tokens: {output_tokens:,}")
    print(f"\nTime: {elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
