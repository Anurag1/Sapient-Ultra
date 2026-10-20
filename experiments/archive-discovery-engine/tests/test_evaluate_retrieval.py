import unittest

from evaluate_retrieval import evaluate, ndcg_at_k, precision_at_k, recall_at_k


class EvaluationTests(unittest.TestCase):
    def test_basic_metrics(self):
        ranked = ["a", "b", "c"]
        relevant = {"a", "c"}
        self.assertAlmostEqual(precision_at_k(ranked, relevant, 2), 0.5)
        self.assertAlmostEqual(recall_at_k(ranked, relevant, 2), 0.5)
        self.assertAlmostEqual(ndcg_at_k(["a", "c"], {"a": 2, "c": 1}, 2), 1.0)

    def test_precision_requires_positive_k(self):
        with self.assertRaises(ValueError):
            precision_at_k([], set(), 0)

    def test_unknown_qrels_id_fails_closed(self):
        corpus = [{"id": "a", "title": "Alpha", "text": "one", "source_url": "https://example.org/a", "entities": []}]
        with self.assertRaises(ValueError):
            evaluate(corpus, {"queries": {"alpha": {"missing-id": 2}}}, 1)

    def test_evaluate_returns_both_methods(self):
        corpus = [
            {"id": "a", "title": "Ada mathematics", "text": "calculations", "source_url": "https://example.org/a", "entities": ["Ada"]},
            {"id": "b", "title": "Babbage computing", "text": "engine", "source_url": "https://example.org/b", "entities": ["Ada", "Babbage"]},
        ]
        result = evaluate(corpus, {"queries": {"Ada mathematics": {"a": 2, "b": 1}}}, 2)
        self.assertEqual(set(result["methods"]), {"keyword", "graph"})
        self.assertEqual(result["query_count"], 1)


if __name__ == "__main__":
    unittest.main()
