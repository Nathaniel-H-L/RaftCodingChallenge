# Raft Principle Machine Learning Engineer Coding Challenge

## Notes
1. To run you must update the .env with your OPENROUTER_API_KEY, this can be generated at: https://openrouter.ai/settings/keys
2. I wrote this to be able to run as an installable package, or in CLI
    - Installable package: From the root directory, run `python -m pip install .` to install, then run `raft-agent "Natural Language prompt in quotes."`
    - CLI: From the folder "raft_agent" run `python main.py "Natural Language prompt in quotes."`
3. I also included BOTH a setup.py and pyproject.toml for easy install and backwards compatability.

## Overview
This project implements a **deterministic, production-oriented AI agent** that converts natural language queries into structured JSON results over unstructured order data.

The system treats the LLM as a **constrained parsing component**, not a decision-maker. All business logic, filtering, and validation are handled deterministically in code to ensure correctness and robustness.

---

## Problem Statement

Given:
- A dummy customer API returning **unstructured text** about orders
- A natural language user query (e.g. _“Show me all orders where the buyer was located in Ohio and total value was over 500”_)

The system must:
1. Parse the user’s intent
2. Fetch raw order data via API
3. Extract structured fields using an LLM
4. Apply deterministic filtering
5. Return clean JSON output

---

## Key Design Principles

### 1. LLMs are parsers, not authorities

The LLM is used **only** for:
- Extracting intent from natural language
- Extracting explicit fields from unstructured text

The LLM is **never allowed** to:
- Filter orders
- Compare numeric values
- Infer missing fields
- Decide which orders qualify

All decisions are made deterministically in Python.

---

### 2. Determinism over cleverness

To ensure correctness and repeatability:
- `temperature=0`
- Strict JSON schemas enforced via Pydantic
- Missing or ambiguous fields resolve to `null`
- Invalid outputs are dropped, not repaired by the model

This prevents hallucinations from propagating downstream.

---

### 3. Robustness to unstructured and changing data

The API response is treated as opaque text:
- No assumptions about schema stability
- No reliance on fixed keys
- Orders are extracted via schema-constrained parsing

This makes the system resilient to:
- Field reordering
- Missing keys
- Textual format changes

---

### 4. Context-window safety

Raw API responses are chunked before LLM processing to:
- Prevent context window overflow
- Enable scaling to larger datasets
- Avoid single-prompt failure modes

---

## Architecture

### High-level flow
```markdown
User Query
   |
   v
[Intent Parser (LLM, constrained)]
   |
   v
[Orders API Fetcher]
   |
   v
[Chunking / Preprocessing]
   |
   v
[Order Field Extractor (LLM, schema-enforced)]
   |
   v
[Deterministic Validation & Filtering]
   |
   v
Structured JSON Output