---
title: Recursive Language Model (RLM)
description: A Microsoft Agent Framework implementation of Recursive Language Models for processing unbounded-length context with LLMs
author: Corrado Cavalli
ms.date: 2026-03-24
ms.topic: overview
keywords:
  - recursive language model
  - rlm
  - microsoft agent framework
  - long context
  - python repl
---

## Overview

This repository contains a Python implementation of Recursive Language Models (RLM), based on the
[research paper by Alex Zhang and Omar Khattab (MIT, 2025)](https://github.com/ysz/recursive-llm).
The implementation uses the [Microsoft Agent Framework](https://pypi.org/project/agent-framework/)
to orchestrate LLM calls and tool execution.

RLM solves a fundamental limitation of large language models: processing documents that exceed
the model's effective context window. Instead of stuffing the entire document into the prompt,
RLM stores the context as a variable and gives the LLM a Python REPL tool to explore, extract,
and analyze it programmatically.

## How RLM Works

Traditional LLM usage passes the full document directly in the prompt. This approach degrades in
accuracy as context length grows, since the model must attend to the entire document at once. RLM
takes a different approach:

1. The context (document) is stored as a Python variable, not included in the system prompt.
2. The LLM receives a `execute_python` tool that runs code in a sandboxed REPL environment.
3. The REPL provides access to `context` (the document), `query` (the question), standard modules
   (`re`, `json`, `math`), and common utilities (`Counter`, `defaultdict`).
4. The LLM explores the document incrementally: first scanning structure, then extracting data
   with regex, then aggregating results across multiple tool calls.
5. For complex analysis, a `recursive_llm(sub_query, sub_context)` function is available in the
   REPL. This spawns a new RLM agent at a deeper recursion level to process a sub-section of the
   document, enabling divide-and-conquer strategies.
6. Recursion depth is capped (default: 5 levels) to prevent runaway chains.

This approach lets the model work with arbitrarily large documents while keeping each LLM call
within a manageable prompt size. The model effectively "programs" its own analysis rather than
relying on passive attention over long text.

## Repository Structure

```text
rlm/
├── rlm_maf/                  # Core library
│   ├── __init__.py            # Public API exports (RLM, RLMStats, REPLExecutor)
│   ├── core.py                # RLM class, recursive agent orchestration, token tracking
│   ├── prompts.py             # System prompt template for the REPL-based agent
│   └── repl.py                # Sandboxed Python REPL executor with whitelisted builtins
├── data/                      # Sample documents for testing
│   ├── ai_history.txt         # Comprehensive AI history document (~long-form)
│   ├── nexus_finance_q3_2025.txt
│   ├── nexus_hr_q3_2025.txt
│   ├── nexus_operations_q3_2025.txt
│   ├── nexus_rnd_q3_2025.txt
│   └── nexus_strategy_q3_2025.txt
├── basic_usage.py             # Example: query a document using RLM
├── basic_usage_no_rlm.py      # Example: same query without RLM (full context in prompt)
├── verify.py                  # Ground truth verification script for ai_history.txt
├── .env.example               # Template for required environment variables
├── pyproject.toml             # Project metadata and dependencies
└── README.md
```

## Prerequisites

* Python 3.13 or later
* [uv](https://docs.astral.sh/uv/) package manager
* An Azure OpenAI deployment with the Responses API enabled
* Azure CLI authenticated (`az login`)

## Setup

1. Clone the repository and navigate to the project directory:

   ```bash
   git clone <repository-url>
   cd rlm
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   uv sync
   ```

3. Copy the example environment file and fill in your Azure OpenAI credentials:

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` with your values:

   ```text
   AZURE_AI_PROJECT_ENDPOINT=https://<your-resource>.services.ai.azure.com/api/projects/<your-project>
   AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME=<your-deployment-name>
   ```

4. Ensure you are logged into Azure CLI:

   ```bash
   az login
   ```

## Usage

### Running with RLM

The `basic_usage.py` script demonstrates RLM processing a complex, multi-part analytical query
against the `ai_history.txt` document:

```bash
uv run basic_usage.py
```

The script loads the document, creates an RLM instance, and lets the agent explore the text
through multiple REPL calls. After completion, it prints the answer along with token usage
statistics (LLM calls, input/output tokens, max recursion depth).

### Running without RLM (baseline comparison)

The `basic_usage_no_rlm.py` script sends the same query with the full document embedded directly
in the prompt, providing a baseline for comparison:

```bash
uv run basic_usage_no_rlm.py
```

### Verifying results

The `verify.py` script computes ground truth answers for the sample queries using deterministic
Python code:

```bash
uv run verify.py
```

## Library API

```python
from rlm_maf import RLM

rlm = RLM(client=azure_client, max_depth=5)
result = await rlm.complete(query="Your question", context="Your document text")

# Access token usage statistics
print(rlm.stats.llm_calls)
print(rlm.stats.total_input_tokens)
print(rlm.stats.total_output_tokens)
print(rlm.stats.max_depth_reached)
```

## Dependencies

* `agent-framework` (>= 1.0.0rc5): Microsoft Agent Framework for LLM orchestration and tool execution.
* `python-dotenv` (>= 1.2.2): Loads environment variables from `.env` files.
* `rich` (>= 14.3.3): Terminal formatting utilities.

