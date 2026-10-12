
import tempfile
import unittest
from pathlib import Path

from document_loader import (
    load_text_files,
    chunk_text,
    build_chunks,
)


class TestDocumentLoader(unittest.TestCase):

    def test_empty_text(self):
        self.assertEqual(chunk_text(""), [])

    def test_short_text(self):
        self.assertEqual(
            chunk_text("Hello world", chunk_size=20, overlap=2),
            ["Hello world"],
        )

    def test_long_text_is_split(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = chunk_text(
            text,
            chunk_size=10,
            overlap=2,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunks))

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("example", chunk_size=10, overlap=10)

    def test_load_and_build_chunks(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "notes.txt"
            path.write_text(
                "Python supports AI development.",
                encoding="utf-8",
            )

            docs = load_text_files(temp)
            chunks = build_chunks(docs)

            self.assertEqual(len(docs), 1)
            self.assertEqual(chunks[0]["source"], "notes.txt")
            self.assertIn("Python", chunks[0]["text"])


if __name__ == "__main__":
    unittest.main()
