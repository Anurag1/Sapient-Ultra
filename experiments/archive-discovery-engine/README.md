# Archive Discovery Engine (prototype)

A dependency-free, reproducible experiment for evidence-linked discovery across a small corpus inspired by Internet Archive research workflows. It compares lexical keyword ranking with transparent graph-assisted ranking using explicitly supplied entity links.

> **Scope and honesty:** This is a local proof of concept using synthetic demonstration records and Internet Archive search URLs. It is not a live crawl, not an Internet Archive product, and not a factuality/contradiction detector. Scores are ranking scores, not probabilities. Evaluate against permitted real metadata/OCR and human-labelled relevance judgments before claiming improvement.

## Run locally

Requires Python 3.10+; no third-party packages are needed.

```bash
cd experiments/archive-discovery-engine
python archive_discovery.py
python archive_discovery.py --query "Charles Babbage Analytical Engine" --top-k 5
python archive_discovery.py --json
python -m unittest discover -s tests -v
```

## What it does

1. Loads documents with stable IDs, text, source URLs, and declared entities.
2. Ranks documents with a lexical baseline.
3. Adds an explainable boost when a query names a relatively distinctive entity and retrieves related documents through a one-hop entity relationship.
4. Returns source URLs, evidence text, declared entities, and explanations.
5. Tests ranking, provenance fields, entity links, empty queries, duplicate IDs, and comparison output.

## Demo data

`sample_corpus.json` is synthetic teaching data. Its `archive.org/search` links are search entry points, **not citations proving that the demo passage appears in a specific archived item**. Replace this data with permitted item metadata and OCR passages whose item IDs and page/section references have been verified.

## Evaluation plan

- Build a permitted corpus with item identifiers and page/section provenance.
- Have reviewers label query-to-passage relevance.
- Compare keyword and graph-assisted retrieval using Precision@k, Recall@k, and nDCG@k.
- Track missing/incorrect provenance and false entity links.
- Publish sample size, test queries, limitations, and reproducible commands.

## Responsible-use constraints

Respect each item's rights, access conditions, and Internet Archive policies. Preserve source provenance. Treat entity links and potential disagreements as reviewable leads, not verified facts. AI-generated summaries do not replace the source.

## Relationship to BookReader

This is intentionally a standalone research prototype, not a patch to `internetarchive/bookreader`. A future integration should follow BookReader's contribution process and preserve accessibility and compatibility requirements.
