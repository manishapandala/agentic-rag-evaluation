"""Deterministic agentic retrieval with inspectable planning and verification."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import re
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a", "after", "an", "and", "are", "as", "at", "be", "by", "can", "do",
    "does", "during", "for", "from", "how", "i", "in", "is", "it", "makes",
    "of", "on", "or", "should", "the", "to", "what", "when", "why", "with",
}
TERM_ALIASES = {
    "answers": "answer",
    "backtested": "backtest",
    "backtests": "backtest",
    "calls": "call",
    "citations": "cite",
    "citation": "cite",
    "compute": "calculate",
    "evaluation": "evaluate",
    "forecasts": "forecast",
    "orders": "sales",
    "resources": "resource",
    "revenue": "sales",
    "tools": "tool",
    "traces": "trace",
    "vendor": "supplier",
}
SYNONYMS = {
    "hallucination": ("grounded", "citation", "evidence"),
    "reliable": ("calibration", "drift", "monitoring"),
    "sales": ("revenue", "orders", "forecast"),
    "supplier": ("vendor", "third", "party"),
    "image": ("vision", "pixels", "geometry"),
    "tools": ("tool", "function", "trace"),
}


def tokenize(text: str) -> list[str]:
    tokens = [token for token in TOKEN_RE.findall(text.lower()) if token not in STOP_WORDS]
    return [TERM_ALIASES.get(token, token) for token in tokens]


@dataclass(frozen=True)
class Document:
    document_id: str
    topic: str
    title: str
    text: str


@dataclass(frozen=True)
class RetrievedDocument:
    document_id: str
    topic: str
    title: str
    score: float
    text: str


@dataclass(frozen=True)
class AgentResult:
    query: str
    answer: str
    citations: tuple[str, ...]
    evidence_coverage: float
    retrieved: tuple[RetrievedDocument, ...]
    trace: tuple[dict, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["citations"] = list(self.citations)
        data["retrieved"] = [asdict(item) for item in self.retrieved]
        data["trace"] = list(self.trace)
        return data


class SearchIndex:
    """Small BM25-style index used as an auditable retrieval baseline."""

    def __init__(self, documents: Iterable[Document]) -> None:
        self.documents = list(documents)
        if not self.documents:
            raise ValueError("At least one document is required")
        self.tokens = [tokenize(document.title + " " + document.text) for document in self.documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths)
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        total = len(self.documents)
        self.idf = {
            token: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def search(self, query: str, limit: int = 3) -> list[RetrievedDocument]:
        if limit < 1:
            raise ValueError("limit must be positive")
        query_tokens = tokenize(query)
        ranked: list[RetrievedDocument] = []
        for document, tokens, length in zip(self.documents, self.tokens, self.lengths, strict=True):
            frequencies = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies[token]
                if not frequency:
                    continue
                denominator = frequency + 1.2 * (0.25 + 0.75 * length / self.average_length)
                score += self.idf.get(token, 0.0) * (frequency * 2.2 / denominator)
            ranked.append(
                RetrievedDocument(
                    document_id=document.document_id,
                    topic=document.topic,
                    title=document.title,
                    score=round(score, 4),
                    text=document.text,
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.document_id))
        selected: list[RetrievedDocument] = []
        seen_topics: set[str] = set()
        for item in ranked:
            if item.topic in seen_topics:
                continue
            selected.append(item)
            seen_topics.add(item.topic)
            if len(selected) == limit:
                return selected
        for item in ranked:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) == limit:
                break
        return selected


class RetrievalAgent:
    def __init__(self, documents: Iterable[Document], coverage_threshold: float = 0.55) -> None:
        self.index = SearchIndex(documents)
        self.coverage_threshold = coverage_threshold

    @staticmethod
    def _expand(query: str) -> str:
        additions: list[str] = []
        for token in tokenize(query):
            additions.extend(SYNONYMS.get(token, ()))
        return " ".join([query, *additions]).strip()

    @staticmethod
    def _coverage(query: str, retrieved: list[RetrievedDocument]) -> float:
        query_terms = set(tokenize(query))
        if not query_terms:
            return 0.0
        evidence_terms = set(tokenize(" ".join(item.text for item in retrieved)))
        return round(len(query_terms & evidence_terms) / len(query_terms), 3)

    @staticmethod
    def _best_sentence(query: str, document: RetrievedDocument) -> str:
        query_terms = set(tokenize(query))
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", document.text) if part.strip()]
        return max(sentences, key=lambda sentence: (len(query_terms & set(tokenize(sentence))), -len(sentence)))

    def run(self, query: str) -> AgentResult:
        if not query.strip():
            raise ValueError("query cannot be empty")
        trace: list[dict] = [{"step": "plan", "detail": "retrieve, verify evidence, then synthesize"}]
        retrieved = self.index.search(query, limit=3)
        trace.append({"step": "retrieve", "query": query, "document_ids": [item.document_id for item in retrieved]})
        coverage = self._coverage(query, retrieved)
        trace.append({"step": "verify", "evidence_coverage": coverage, "threshold": self.coverage_threshold})

        expanded_query = self._expand(query)
        if coverage < self.coverage_threshold and expanded_query != query:
            retry = self.index.search(expanded_query, limit=3)
            retry_coverage = self._coverage(query, retry)
            trace.append({"step": "rewrite", "expanded_query": expanded_query})
            trace.append({"step": "retrieve_again", "document_ids": [item.document_id for item in retry]})
            if retry_coverage >= coverage:
                retrieved, coverage = retry, retry_coverage

        useful = [item for item in retrieved if item.score > 0]
        if not useful:
            trace.append({"step": "stop", "reason": "no supporting evidence"})
            return AgentResult(query, "Insufficient evidence in the indexed corpus.", (), 0.0, tuple(retrieved), tuple(trace))
        if coverage < self.coverage_threshold:
            trace.append({"step": "stop", "reason": "evidence coverage below threshold"})
            return AgentResult(
                query,
                "Insufficient evidence coverage to answer this question.",
                (),
                coverage,
                tuple(retrieved),
                tuple(trace),
            )

        evidence = [self._best_sentence(query, item) for item in useful[:2]]
        citations = tuple(item.document_id for item in useful[:2])
        answer = " ".join(evidence) + " " + " ".join(f"[{citation}]" for citation in citations)
        trace.append({"step": "synthesize", "citations": list(citations), "grounding_rule": "retrieved sentences only"})
        return AgentResult(query, answer, citations, coverage, tuple(retrieved), tuple(trace))
