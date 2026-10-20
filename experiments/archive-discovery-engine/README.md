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


## Milestone 2: live Internet Archive ingestion

The new `live_archive.py` module performs **read-only** retrieval from public Internet Archive endpoints:

- Advanced Search API for item identifiers and descriptive metadata.
- Item Metadata API for the canonical item metadata and file list.
- Public OCR/plain-text files when present and within the configured size limit.
- Passage records preserve the item identifier, item URL, source file URL, OCR locator, creator/subjects, date, and rights metadata when supplied.

The ingestion intentionally does not write to Archive.org, bypass access restrictions, or download arbitrary binary files. It caps each search to 25 items and each OCR file to 3 MB. Items without a suitable public text file are skipped. OCR text can contain errors.

### Run a live ingest

From this directory, with Python 3.10+ and an internet connection:

```bash
python live_archive.py --query 'title:mathematics AND mediatype:texts' --rows 5 --out live_archive_corpus.json
```

The query syntax is the Internet Archive Advanced Search query language. Start with a small `--rows` value. The program prints counts and failures and writes the corpus JSON locally.

Passages split on explicit form-feed characters are labelled `text-page-N`; other passages are labelled `text-segment-N`. **Text segment numbers are not physical page numbers.** Verify any citation in the original item before using it in research or publication.

### Retrieval evaluation

`evaluate_retrieval.py` compares keyword and graph-assisted rankings using Precision@k, Recall@k, and nDCG@k. It requires explicit relevance labels and fails if labels refer to IDs absent from the corpus.

1. Run a live ingest.
2. Inspect the corpus and verify the item/file provenance.
3. Copy `qrels.template.json` to `qrels.json`.
4. Add several representative queries and human-judged relevance grades for exact passage IDs (0 = irrelevant, 1 = partly relevant, 2 = highly relevant).
5. Run:

```bash
python evaluate_retrieval.py --corpus live_archive_corpus.json --qrels qrels.json --k 5 --out retrieval_evaluation.json
```

Do not treat the placeholder template as evaluation data. Do not claim an improvement from a tiny or unrepresentative set of queries. Report query count, relevance-labeling method, skipped items, provenance errors, and both methods' metrics.

### Validation

```bash
python -m unittest discover -s tests -v
```

The tests cover the new parsing, source provenance, safety limits, and metric calculations. Network-dependent live ingestion should be run separately because availability and OCR coverage vary by item.
