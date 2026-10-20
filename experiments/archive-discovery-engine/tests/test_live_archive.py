import unittest
from unittest.mock import patch

import live_archive


class LiveArchiveTests(unittest.TestCase):
    def test_search_rejects_empty_query(self):
        with self.assertRaises(ValueError):
            live_archive.search_items("  ")

    def test_search_caps_rows(self):
        with self.assertRaises(ValueError):
            live_archive.search_items("history", rows=26)

    def test_passages_use_explicit_page_breaks(self):
        pages = live_archive._passages("page one\fpage two")
        self.assertEqual(pages, [("text-page-1", "page one"), ("text-page-2", "page two")])

    def test_passages_do_not_invent_physical_page_numbers(self):
        passages = live_archive._passages("paragraph one\n\nparagraph two")
        self.assertTrue(all(locator.startswith("text-segment-") for locator, _ in passages))

    def test_prefers_canonical_djvu_text(self):
        chosen = live_archive._text_file([
            {"name": "notes.txt", "format": "Plain Text", "size": "100"},
            {"name": "book_djvu.txt", "format": "DjVuTXT", "size": "100"},
        ])
        self.assertEqual(chosen["name"], "book_djvu.txt")

    def test_skips_private_or_oversized_files(self):
        chosen = live_archive._text_file([
            {"name": "private_djvu.txt", "format": "DjVuTXT", "size": "100", "private": True},
            {"name": "huge_djvu.txt", "format": "DjVuTXT", "size": str(live_archive.MAX_TEXT_BYTES + 1)},
        ])
        self.assertIsNone(chosen)

    @patch("live_archive._read_json", return_value={"metadata": {"title": "A Book", "creator": "A. Author", "subject": ["History"]}, "files": [{"name": "a_djvu.txt", "format": "DjVuTXT", "size": "15"}]})
    @patch("live_archive._read_url", return_value=b"First OCR page\fSecond OCR page")
    def test_fetch_passages_keep_item_file_and_locator(self, _read_url, _read_json):
        docs = live_archive.fetch_item_passages({"identifier": "valid_item"}, delay_seconds=0)
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]["item_identifier"], "valid_item")
        self.assertEqual(docs[0]["passage_locator"], "text-page-1")
        self.assertEqual(docs[0]["locator_type"], "ocr-page-boundary")
        self.assertIn("archive.org/download/valid_item/a_djvu.txt", docs[0]["source_file_url"])
        self.assertIn("A. Author", docs[0]["entities"])

    def test_invalid_identifier_rejected(self):
        with self.assertRaises(ValueError):
            live_archive.fetch_item_passages({"identifier": "../private"}, delay_seconds=0)


if __name__ == "__main__":
    unittest.main()
