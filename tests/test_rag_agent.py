import unittest

from rag_agent import Document, RetrievalAgent, SearchIndex, tokenize


DOCUMENTS = [
    Document("D1", "rag", "Grounded RAG", "Grounded answers use retrieval evidence and citations."),
    Document("D2", "agents", "Agent traces", "Agent traces record tool calls, plans, retries, and guardrails."),
    Document("D3", "forecasting", "Forecast evaluation", "Forecasts use chronological backtests and seasonal baselines."),
]


class RetrievalAgentTests(unittest.TestCase):
    def test_tokenize_normalizes_and_removes_stop_words(self):
        self.assertEqual(tokenize("How are the Grounded Answers?"), ["grounded", "answer"])

    def test_search_returns_best_topic(self):
        result = SearchIndex(DOCUMENTS).search("retrieval citations")[0]
        self.assertEqual(result.topic, "rag")

    def test_search_rejects_invalid_limit(self):
        with self.assertRaises(ValueError):
            SearchIndex(DOCUMENTS).search("test", limit=0)

    def test_empty_index_is_rejected(self):
        with self.assertRaises(ValueError):
            SearchIndex([])

    def test_agent_returns_citations_and_trace(self):
        result = RetrievalAgent(DOCUMENTS).run("How do grounded answers use citations?")
        self.assertEqual(result.citations[0], "D1")
        self.assertEqual(result.trace[-1]["step"], "synthesize")

    def test_agent_refuses_empty_query(self):
        with self.assertRaises(ValueError):
            RetrievalAgent(DOCUMENTS).run("   ")

    def test_agent_stops_without_evidence(self):
        result = RetrievalAgent(DOCUMENTS).run("quantum entanglement")
        self.assertEqual(result.citations, ())
        self.assertIn("Insufficient evidence", result.answer)

    def test_agent_stops_when_evidence_coverage_is_too_low(self):
        result = RetrievalAgent(DOCUMENTS, coverage_threshold=0.9).run(
            "grounded answers plus unrelated quantum details"
        )
        self.assertEqual(result.citations, ())
        self.assertEqual(result.trace[-1]["reason"], "evidence coverage below threshold")

    def test_result_is_json_ready(self):
        result = RetrievalAgent(DOCUMENTS).run("agent tool traces").to_dict()
        self.assertIsInstance(result["retrieved"], list)
        self.assertIsInstance(result["trace"], list)


if __name__ == "__main__":
    unittest.main()
