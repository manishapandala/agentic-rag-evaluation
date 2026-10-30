# Agentic RAG Evaluation Lab

A provider-neutral retrieval agent that plans, retrieves, verifies evidence coverage, optionally rewrites a query, and synthesizes a citation-bound answer. The repository focuses on the engineering work around an agent: structured traces, deterministic evaluation, grounded outputs, and explicit refusal when evidence is missing.

## Why this project

Modern agent systems need more than a prompt. Tool execution, guardrails, tracing, handoffs, and evaluation are core production concerns. This project implements those ideas with a local deterministic retrieval baseline, so every published number can be regenerated without an API key.

## Run

```bash
python -m pip install -r requirements-dev.txt
ruff check .
python -m unittest discover -s tests -v
python run_demo.py
```

See [ENGINEERING.md](ENGINEERING.md) for the component boundaries, evaluation contract, failure handling, and extension path.

## Outputs

- `demo_summary.json` — top-1 topic accuracy and local latency for eight generated-corpus queries;
- `sample_trace.json` — one complete plan → retrieve → verify → synthesize trace with citations;
- the project page — a visual trace and downloadable copies of both artifacts.

## Scope

The corpus is generated and the answer composer is deterministic. Results validate orchestration and evaluation contracts, not the quality of a hosted LLM or a production knowledge base. A next step would compare providers while keeping the same trace and evaluation schema.

## Repository map

- `rag_agent.py` — retrieval index, orchestration loop, evidence verification, and citation-bound synthesis;
- `run_demo.py` — deterministic corpus generation and evaluation harness;
- `tests/` — behavioral tests covering routing, grounding, refusal, and trace structure;
- `index.html` — GitHub Pages case study with downloadable outputs;
- `.github/workflows/tests.yml` — lint, test, and demo regeneration on each push.
