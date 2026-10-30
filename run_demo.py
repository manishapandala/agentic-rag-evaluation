"""Generate a repeatable benchmark and trace artifact for the retrieval agent."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

from rag_agent import Document, RetrievalAgent


KNOWLEDGE = [
    ("rag", "Grounded retrieval evaluation", "Grounded answers cite retrieved evidence. Retrieval evaluation measures hit rate, evidence coverage, and unsupported claims before deployment."),
    ("agents", "Observable agent workflows", "Agent workflows record plans, tool calls, retries, guardrail decisions, and final citations in an inspectable trace."),
    ("analytics", "Guarded analytics tools", "A data analyst agent should validate the schema, allowlist tools, compute metrics, and preserve an audit trail for every result."),
    ("security", "Zero trust supplier access", "Zero trust limits vendor and supplier access through continuous verification, least privilege, and session monitoring."),
    ("forecasting", "Demand forecasting backtests", "Revenue and order forecasts require chronological backtests, seasonal baselines, and prediction intervals."),
    ("reliability", "Reliable model monitoring", "Reliable machine learning separates calibration, threshold selection, drift monitoring, and final test evaluation."),
    ("vision", "Geometry-aware image inspection", "Computer vision inspection preserves image pixels and geometry while ranking unusual battery structures for review."),
    ("mcp", "Model Context Protocol tools", "MCP connects AI applications to typed tools, read-only resources, and reusable prompts through a standard protocol."),
]

QUERIES = [
    ("How do I evaluate grounded RAG answers and citations?", "rag"),
    ("What should an agent trace record for tool calls?", "agents"),
    ("How can a data agent safely calculate business metrics?", "analytics"),
    ("How does zero trust reduce supplier access risk?", "security"),
    ("How should sales demand forecasts be backtested?", "forecasting"),
    ("What makes a machine learning model reliable after deployment?", "reliability"),
    ("Why preserve image geometry during anomaly inspection?", "vision"),
    ("How can AI applications connect to typed tools and resources?", "mcp"),
]


def build_documents(repeats: int = 10) -> list[Document]:
    documents = []
    for repeat in range(repeats):
        for index, (topic, title, text) in enumerate(KNOWLEDGE):
            documents.append(Document(f"DOC-{repeat * len(KNOWLEDGE) + index:03d}", topic, f"{title} {repeat + 1}", f"{text} Evaluation note {repeat + 1} keeps this generated corpus deterministic."))
    return documents


def main() -> None:
    agent = RetrievalAgent(build_documents())
    durations = []
    results = []
    correct = 0
    for query, expected_topic in QUERIES:
        started = time.perf_counter()
        result = agent.run(query)
        durations.append((time.perf_counter() - started) * 1000)
        top_topic = result.retrieved[0].topic
        correct += top_topic == expected_topic
        results.append({"query": query, "expected_topic": expected_topic, "top_topic": top_topic, "evidence_coverage": result.evidence_coverage, "citations": list(result.citations)})

    sample = agent.run("How should an agent evaluate grounded answers and record tool traces?")
    summary = {
        "scope": "deterministic generated-corpus agent evaluation",
        "documents": len(agent.index.documents),
        "evaluation_queries": len(QUERIES),
        "top_1_topic_accuracy": round(correct / len(QUERIES), 4),
        "median_run_ms": round(statistics.median(durations), 4),
        "p95_run_ms": round(sorted(durations)[int(0.95 * (len(durations) - 1))], 4),
        "results": results,
    }
    Path("demo_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    Path("sample_trace.json").write_text(json.dumps(sample.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
