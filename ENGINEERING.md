# Engineering notes

## Architecture

The project separates the retrieval index, orchestration policy, evidence checks, answer synthesis, and evaluation harness. `SearchIndex` owns token statistics and ranking. `RAGAgent` owns the plan → retrieve → verify → optionally rewrite → synthesize loop. The demo harness supplies generated evidence and scores observable behavior without changing either component.

## Evaluation contract

- Every answer must cite the identifiers of retrieved documents.
- No-evidence queries must return a refusal instead of an invented answer.
- Query expansion is visible in the trace and bounded to one retry.
- Published accuracy is measured on eight deterministic queries over an 80-document generated corpus.
- Latency is local and machine-specific; it is not presented as a production service-level claim.

## Failure handling

The agent returns structured status values for answered and refused requests. Each run records the original query, plan, searches, verification results, citations, and final status. This makes a failed evaluation inspectable instead of reducing it to a single score.

## Extension path

A hosted model can replace deterministic synthesis, and an MCP or vector-database adapter can replace local retrieval. Those changes should preserve the trace schema, refusal behavior, bounded retry policy, and evaluation fixtures so provider changes remain comparable.
