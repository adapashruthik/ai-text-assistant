
import unittest

from rag_core import build_prompt


class TestRAGCore(unittest.TestCase):

    def test_prompt_contains_question(self):
        docs = [{
            "id": "docker",
            "title": "Docker",
            "text": "Docker runs applications in containers.",
        }]

        system, user = build_prompt("What is Docker?", docs)

        self.assertIn("What is Docker?", user)
        self.assertIn("reference material", system.lower())

    def test_prompt_contains_source(self):
        docs = [{
            "id": "python",
            "title": "Python",
            "text": "Python is a programming language.",
        }]

        _, user = build_prompt("What is Python?", docs)

        self.assertIn("python", user.lower())
        self.assertIn("Python is a programming language.", user)

    def test_empty_question(self):
        with self.assertRaises(ValueError):
            build_prompt("   ", [{
                "id": "doc1",
                "title": "Example",
                "text": "Example content",
            }])

    def test_empty_context(self):
        with self.assertRaises(ValueError):
            build_prompt("What is Docker?", [])


if __name__ == "__main__":
    unittest.main()
