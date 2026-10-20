"""Evaluate retrieval methods against explicit human relevance labels.

qrels JSON shape:
{
  "queries": {
    "query text": {"relevant_document_id": 2, "another_id": 1}
  }
}
Grades are non-negative integers; 0 means not relevant. Do not report metrics
until relevance labels have been independently reviewed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

from archive_discovery import graph_search, keyword_search, load_corpus


def precision_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(doc_id in relevant for doc_id in ranked_ids[:k]) / k


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    return sum(doc_id in relevant for doc_id in ranked_ids[:k]) / len(relevant)


def _dcg(ranked_ids: list[str], grades: dict[str, int], k: int) -> float:
    return sum((2 ** grades.get(doc_id, 0) - 1) / math.log2(rank + 2)
               for rank, doc_id in enumerate(ranked_ids[:k]))


def ndcg_at_k(ranked_ids: list[str], grades: dict[str, int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    actual = _dcg(ranked_ids, grades, k)
    ideal_ids = sorted(grades, key=lambda doc_id: (-grades[doc_id], doc_id))
    ideal = _dcg(ideal_ids, grades, k)
    return actual / ideal if ideal else 0.0


def evaluate(corpus: list[dict[str, Any]], qrels: dict[str, Any], k: int = 5) -> dict[str, Any]:
    queries = qrels.get("queries", {})
    if not isinstance(queries, dict) or not queries:
        raise ValueError("qrels must contain a non-empty 'queries' object")
    methods: dict[str, Callable[..., list[dict[str, Any]]]] = {
        "keyword": keyword_search,
        "graph": graph_search,
    }
    per_query: dict[str, Any] = {}
    totals = {name: {"precision_at_k": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0} for name in methods}
    for query, grade_map in queries.items():
        if not isinstance(grade_map, dict):
            raise ValueError(f"Relevance grades must be an object for query: {query}")
        grades: dict[str, int] = {}
        for doc_id, grade in grade_map.items():
            if not isinstance(grade, int) or grade < 0:
                raise ValueError(f"Relevance grade must be a non-negative integer: {query}/{doc_id}")
            grades[str(doc_id)] = grade
        known_ids = {doc["id"] for doc in corpus}
        unknown = set(grades) - known_ids
        if unknown:
            raise ValueError(f"qrels reference unknown document IDs: {sorted(unknown)}")
        relevant = {doc_id for doc_id, grade in grades.items() if grade > 0}
        per_query[query] = {}
        for method_name, method in methods.items():
            ranked = method(query, corpus, k)
            ranked_ids = [item["id"] for item in ranked]
            metrics = {
                "precision_at_k": precision_at_k(ranked_ids, relevant, k),
                "recall_at_k": recall_at_k(ranked_ids, relevant, k),
                "ndcg_at_k": ndcg_at_k(ranked_ids, grades, k),
                "ranked_ids": ranked_ids,
            }
            per_query[query][method_name] = metrics
            for metric in totals[method_name]:
                totals[method_name][metric] += metrics[metric]
    count = len(queries)
    means = {name: {metric: round(value / count, 4) for metric, value in metrics.items()}
             for name, metrics in totals.items()}
    return {
        "k": k,
        "query_count": count,
        "labelled_query_count": count,
        "methods": means,
        "per_query": per_query,
        "warning": "These scores are meaningful only if qrels are human-labelled and representative; they do not establish general superiority.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="JSON corpus array or live ingestion JSON object")
    parser.add_argument("--qrels", required=True, help="Human-labelled query relevance JSON")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default="retrieval_evaluation.json")
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be positive")
    raw = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    corpus = raw.get("documents", []) if isinstance(raw, dict) else raw
    qrels = json.loads(Path(args.qrels).read_text(encoding="utf-8"))
    result = evaluate(corpus, qrels, args.k)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["methods"], indent=2))
    print(f"Evaluated {result['query_count']} labelled queries at k={args.k}; full report: {args.out}")
    print(result["warning"])


if __name__ == "__main__":
    main()
