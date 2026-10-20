"""Read-only Internet Archive metadata/OCR ingestion.

Uses only public Advanced Search, item metadata, and downloadable text files.
No archive items or metadata are modified. OCR page numbers are reported only
when page boundaries are explicit form-feed separators; otherwise the locator
is honestly marked as a text segment rather than a physical page.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

SEARCH_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata/{identifier}"
DOWNLOAD_URL = "https://archive.org/download/{identifier}/{filename}"
USER_AGENT = "ArchiveDiscoveryEngine/0.2 (read-only research prototype)"
MAX_TEXT_BYTES = 3_000_000
TEXT_FORMATS = {"djvutxt", "plain text", "text", "full text", "ocr text"}
TEXT_SUFFIXES = ("_djvu.txt", "_text.txt", ".txt")


class ArchiveFetchError(RuntimeError):
    """A live API request or archive item could not be read."""


def _read_url(url: str, timeout: float = 20.0, max_bytes: int = MAX_TEXT_BYTES) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    try:
        with urlopen(request, timeout=timeout) as response:
            data = response.read(max_bytes + 1)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise ArchiveFetchError(f"Request failed for {url}: {exc}") from exc
    if len(data) > max_bytes:
        raise ArchiveFetchError(f"Response exceeded {max_bytes} byte safety limit: {url}")
    return data


def _read_json(url: str) -> dict[str, Any]:
    try:
        data = json.loads(_read_url(url).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveFetchError(f"Invalid JSON from {url}") from exc
    if not isinstance(data, dict):
        raise ArchiveFetchError(f"Expected a JSON object from {url}")
    return data


def search_items(query: str, rows: int = 5) -> list[dict[str, Any]]:
    """Search public IA metadata. Rows is deliberately capped to keep demos small."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= rows <= 25:
        raise ValueError("rows must be between 1 and 25")
    params = urlencode([
        ("q", query),
        ("fl[]", "identifier"),
        ("fl[]", "title"),
        ("fl[]", "creator"),
        ("fl[]", "date"),
        ("fl[]", "description"),
        ("fl[]", "subject"),
        ("fl[]", "mediatype"),
        ("rows", str(rows)),
        ("page", "1"),
        ("output", "json"),
    ])
    payload = _read_json(f"{SEARCH_URL}?{params}")
    response = payload.get("response", {})
    docs = response.get("docs", [])
    if not isinstance(docs, list):
        raise ArchiveFetchError("Advanced Search response had no docs list")
    return [doc for doc in docs if isinstance(doc, dict) and doc.get("identifier")]


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(v).strip() for v in values if str(v).strip()]


def _text_file(files: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for file in files:
        name = str(file.get("name", ""))
        fmt = str(file.get("format", "")).strip().lower()
        if file.get("private") in (True, "true", "True"):
            continue
        if fmt in TEXT_FORMATS or name.lower().endswith(TEXT_SUFFIXES):
            try:
                size = int(file.get("size", 0) or 0)
            except (TypeError, ValueError):
                size = 0
            if size <= MAX_TEXT_BYTES:
                # Prefer canonical OCR output over arbitrary text attachments.
                priority = 0 if name.lower().endswith("_djvu.txt") else 1
                candidates.append((priority, name, file))
    return sorted(candidates, key=lambda row: (row[0], row[1]))[0][2] if candidates else None


def _passages(text: str) -> list[tuple[str, str]]:
    """Return (locator, text); form-feed page breaks are explicit, not inferred."""
    cleaned = text.replace("\x00", "").strip()
    if not cleaned:
        return []
    if "\f" in cleaned:
        parts = [part.strip() for part in cleaned.split("\f")]
        return [(f"text-page-{index}", part) for index, part in enumerate(parts, 1) if part]
    # Keep retrieval units bounded, but don't mislabel arbitrary chunks as pages.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    if not paragraphs:
        paragraphs = [cleaned]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) > 1800:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return [(f"text-segment-{index}", chunk) for index, chunk in enumerate(chunks, 1)]


def fetch_item_passages(item: dict[str, Any], delay_seconds: float = 0.15) -> list[dict[str, Any]]:
    """Fetch an item's public metadata and OCR text, preserving exact source locators."""
    identifier = str(item.get("identifier", "")).strip()
    if not identifier or not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", identifier):
        raise ValueError("Invalid or missing Internet Archive identifier")
    metadata_url = METADATA_URL.format(identifier=quote(identifier, safe=""))
    payload = _read_json(metadata_url)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ArchiveFetchError(f"No public metadata found for item {identifier}")
    files = payload.get("files", [])
    if not isinstance(files, list):
        files = []
    text_file = _text_file([f for f in files if isinstance(f, dict)])
    if not text_file:
        return []
    filename = str(text_file.get("name", ""))
    if not filename or "/" in filename or "\\" in filename or filename.startswith("."):
        raise ArchiveFetchError(f"Unsafe OCR filename in item {identifier}")
    if delay_seconds > 0:
        time.sleep(min(delay_seconds, 2.0))
    file_url = DOWNLOAD_URL.format(
        identifier=quote(identifier, safe=""),
        filename=quote(filename, safe=""),
    )
    raw = _read_url(file_url, max_bytes=MAX_TEXT_BYTES)
    text = raw.decode("utf-8", errors="replace")
    title = " ".join(_as_list(metadata.get("title"))) or "Untitled archive item"
    creators = _as_list(metadata.get("creator"))
    subjects = _as_list(metadata.get("subject"))
    entities = list(dict.fromkeys((creators + subjects)[:30]))
    docs = []
    for locator, passage in _passages(text):
        docs.append({
            "id": f"{identifier}::{filename}::{locator}",
            "item_identifier": identifier,
            "title": title,
            "text": passage,
            "entities": entities,
            "source_url": f"https://archive.org/details/{quote(identifier, safe='')}",
            "source_file_url": file_url,
            "passage_locator": locator,
            "locator_type": "ocr-page-boundary" if locator.startswith("text-page-") else "text-segment-no-page-boundary",
            "creator": creators,
            "subjects": subjects,
            "date": metadata.get("date"),
            "rights": metadata.get("rights"),
        })
    return docs


def ingest(query: str, rows: int = 5, delay_seconds: float = 0.15) -> dict[str, Any]:
    results = search_items(query, rows=rows)
    corpus: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for item in results:
        identifier = str(item["identifier"])
        try:
            corpus.extend(fetch_item_passages(item, delay_seconds=delay_seconds))
        except (ArchiveFetchError, ValueError) as exc:
            failures.append({"identifier": identifier, "error": str(exc)})
    return {
        "query": query,
        "source": "Internet Archive public Advanced Search + item metadata + public OCR text",
        "items_found": len(results),
        "passages_ingested": len(corpus),
        "documents": corpus,
        "failures": failures,
        "limitations": [
            "Only public metadata and public OCR text files are read; no archive writes are performed.",
            "Text segment numbers are not physical page numbers unless explicit form-feed boundaries exist.",
            "OCR may contain errors; source item and file URLs are preserved for verification.",
            "Rights/access conditions vary by item; this tool does not bypass restrictions.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Internet Archive metadata query, e.g. 'title:mathematics AND mediatype:texts'")
    parser.add_argument("--rows", type=int, default=5, help="Number of archive items to inspect (1-25)")
    parser.add_argument("--out", default="live_archive_corpus.json", help="Output JSON path")
    parser.add_argument("--delay", type=float, default=0.15, help="Polite delay before each OCR download")
    args = parser.parse_args()
    result = ingest(args.query, args.rows, args.delay)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"Internet Archive query: {result['query']}")
    print(f"Items returned: {result['items_found']}")
    print(f"OCR passages ingested: {result['passages_ingested']}")
    print(f"Items skipped/failed: {len(result['failures'])}")
    print(f"Wrote: {args.out}")
    for failure in result["failures"]:
        print(f"  - {failure['identifier']}: {failure['error']}")


if __name__ == "__main__":
    main()
