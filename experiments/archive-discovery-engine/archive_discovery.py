"""Small, dependency-free prototype for evidence-linked archive discovery.

This deterministic experiment ranks passages using lexical overlap and
explicitly declared entity links. It is not a factuality detector.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]*", re.I)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "was", "were", "with", "how", "what", "when", "where",
}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOPWORDS]


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Corpus must be a JSON list")
    seen: set[str] = set()
    for row in data:
        for field in ("id", "title", "text", "source_url", "entities"):
            if field not in row:
                raise ValueError(f"Document missing required field: {field}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate document id: {row['id']}")
        seen.add(row["id"])
        if not isinstance(row["entities"], list):
            raise ValueError(f"entities must be a list for {row['id']}")
    return data


def _lexical_scores(query: str, docs: list[dict[str, Any]]) -> dict[str, float]:
    query_terms = set(tokenize(query))
    if not query_terms:
        return {d["id"]: 0.0 for d in docs}
    doc_terms = {d["id"]: set(tokenize(d["title"] + " " + d["text"])) for d in docs}
    title_terms = {d["id"]: set(tokenize(d["title"])) for d in docs}
    df = Counter(term for terms in doc_terms.values() for term in query_terms if term in terms)
    n = max(len(docs), 1)
    scores: dict[str, float] = {}
    for doc in docs:
        terms = doc_terms[doc["id"]]
        score = 0.0
        for term in query_terms & terms:
            idf = math.log(1 + n / (1 + df[term]))
            score += idf * (1.5 if term in title_terms[doc["id"]] else 1.0)
        scores[doc["id"]] = score / max(math.sqrt(len(terms)), 1.0)
    return scores


def _result(doc: dict[str, Any], score: float, method: str, reasons: list[str]) -> dict[str, Any]:
    return {
        "id": doc["id"],
        "title": doc["title"],
        "score": round(float(score), 4),
        "method": method,
        "source_url": doc["source_url"],
        "evidence": doc["text"],
        "entities": doc["entities"],
        "why": reasons,
    }


def keyword_search(query: str, docs: list[dict[str, Any]], k: int = 5) -> list[dict[str, Any]]:
    """Baseline: rank documents by lexical overlap only."""
    scores = _lexical_scores(query, docs)
    order = sorted(docs, key=lambda d: (-scores[d["id"]], d["id"]))
    return [_result(d, scores[d["id"]], "keyword", []) for d in order[:max(k, 0)] if scores[d["id"]] > 0]


def build_entity_index(docs: list[dict[str, Any]]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for doc in docs:
        for entity in doc["entities"]:
            normalized = " ".join(tokenize(entity))
            if normalized:
                index[normalized].add(doc["id"])
    return dict(index)


def graph_search(query: str, docs: list[dict[str, Any]], k: int = 5) -> list[dict[str, Any]]:
    """Rank lexical matches and expand one hop through declared entity links.

    Generic high-frequency entities are ignored to limit graph-hub noise.
    Relationships come from corpus annotations; no facts are inferred.
    """
    lexical = _lexical_scores(query, docs)
    q = " ".join(tokenize(query))
    index = build_entity_index(docs)
    matched_entities = [e for e in index if e and e in q and len(index[e]) <= 2]
    id_to_doc = {doc["id"]: doc for doc in docs}
    linked_ids: set[str] = set()
    for entity in matched_entities:
        linked_ids.update(index[entity])
    neighbor_entities: set[str] = set()
    for doc_id in list(linked_ids):
        neighbor_entities.update(" ".join(tokenize(e)) for e in id_to_doc[doc_id]["entities"])
    for entity in neighbor_entities:
        neighbors = index.get(entity, set())
        if entity and len(neighbors) <= 2:
            linked_ids.update(neighbors)
    max_lex = max(lexical.values(), default=0.0) or 1.0
    results = []
    for doc in docs:
        base = lexical[doc["id"]] / max_lex
        reasons = []
        if doc["id"] in linked_ids and lexical[doc["id"]] == 0:
            base += 0.35
            reasons.append("reached through a one-hop, explicitly indexed entity relationship")
        elif doc["id"] in linked_ids:
            base += 0.1
            reasons.append("connected through an explicitly indexed entity relationship")
        if lexical[doc["id"]] > 0:
            reasons.append("lexical match in title or passage")
        if reasons:
            results.append(_result(doc, base, "graph", reasons))
    results.sort(key=lambda r: (-r["score"], r["id"]))
    return results[:max(k, 0)]


def compare(query: str, docs: list[dict[str, Any]], k: int = 5) -> dict[str, Any]:
    return {
        "query": query,
        "keyword_results": keyword_search(query, docs, k),
        "graph_results": graph_search(query, docs, k),
        "note": "Scores are prototype ranking scores, not probabilities or truth judgments.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="sample_corpus.json", help="Path to corpus JSON")
    parser.add_argument("--query", default="How did Ada Lovelace connect mathematics and computing?", help="Search query")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="Print full JSON results")
    args = parser.parse_args()
    results = compare(args.query, load_corpus(args.corpus), args.top_k)
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return
    print("ARCHIVE DISCOVERY ENGINE — LOCAL DEMO")
    print(f"Query: {results['query']}\n")
    for key, label in (("keyword_results", "KEYWORD BASELINE"), ("graph_results", "GRAPH-ASSISTED")):
        print(label)
        if not results[key]:
            print("  (no matches)")
        for rank, item in enumerate(results[key], 1):
            print(f"  {rank}. [{item['score']:.4f}] {item['title']} ({item['id']})")
            print(f"     Source: {item['source_url']}")
            if item["why"]:
                print(f"     Why: {'; '.join(item['why'])}")
        print()
    print(results["note"])


if __name__ == "__main__":
    main()
