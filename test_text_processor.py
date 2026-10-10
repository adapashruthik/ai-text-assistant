
import unittest

from text_processor import clean_text, count_words, get_words


class TestTextProcessor(unittest.TestCase):

    def test_clean_text(self):
        self.assertEqual(
            clean_text("  Hello   AI  "),
            "Hello AI"
        )

    def test_count_words(self):
        self.assertEqual(count_words("AI developers build apps"), 4)

    def test_empty_text(self):
        self.assertEqual(count_words("   "), 0)

    def test_get_words(self):
        self.assertEqual(
            get_words("Hello, AI!"),
            ["hello", "ai"]
        )


if __name__ == "__main__":
    unittest.main()
