import json
import tempfile
import unittest
from pathlib import Path

from archive_discovery import build_entity_index, compare, graph_search, keyword_search, load_corpus

ROOT = Path(__file__).resolve().parents[1]


class ArchiveDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = load_corpus(ROOT / "sample_corpus.json")

    def test_loads_demo_corpus(self):
        self.assertEqual(len(self.docs), 5)
        self.assertEqual(len({d["id"] for d in self.docs}), 5)

    def test_keyword_search_returns_evidence_links(self):
        results = keyword_search("Ada Lovelace mathematics computing", self.docs, 3)
        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "demo-001")
        self.assertTrue(results[0]["source_url"].startswith("https://archive.org/"))
        self.assertTrue(results[0]["evidence"])

    def test_graph_search_explains_entity_link(self):
        results = graph_search("Ada Lovelace", self.docs, 5)
        self.assertTrue(any(r["id"] == "demo-002" for r in results))
        linked = next(r for r in results if r["id"] == "demo-002")
        self.assertTrue(any("entity" in reason for reason in linked["why"]))

    def test_entity_index_is_deterministic(self):
        index = build_entity_index(self.docs)
        self.assertIn("ada lovelace", index)
        self.assertIn("demo-001", index["ada lovelace"])

    def test_empty_query_returns_no_matches(self):
        self.assertEqual(keyword_search("the and of", self.docs), [])
        self.assertEqual(graph_search("the and of", self.docs), [])

    def test_duplicate_ids_are_rejected(self):
        data = [self.docs[0], dict(self.docs[0])]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dup.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate document id"):
                load_corpus(path)

    def test_compare_has_both_methods_and_caveat(self):
        result = compare("Ada Lovelace computing", self.docs)
        self.assertIn("keyword_results", result)
        self.assertIn("graph_results", result)
        self.assertIn("not probabilities", result["note"])


if __name__ == "__main__":
    unittest.main()
