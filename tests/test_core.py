"""Fast invariant checks; no downloads, credentials, or GPU required."""
import math
import unittest
from topic_embedding.cli import parser, validate_args, json_safe
from topic_embedding.preprocessing import clean_text
from topic_embedding.embeddings import POOLING, cache_key, chunk_text
from topic_embedding.evaluation import topic_diversity


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


class CoreTests(unittest.TestCase):
    def test_pooling(self):
        self.assertEqual(len(POOLING), 7)
        self.assertEqual(len(set(POOLING.values())), 7)
        self.assertTrue(all(any(v) for v in POOLING.values()))

    def test_cache_identity(self):
        a = cache_key(["a", "b"], {"pooling": "C"})
        self.assertNotEqual(a, cache_key(["b", "a"], {"pooling": "C"}))
        self.assertNotEqual(a, cache_key(["a", "b"], {"pooling": "M"}))

    def test_chunk_coverage(self):
        chunks, lengths = chunk_text("0 1 2 3 4 5 6", Tokenizer(), 4, 1)
        self.assertEqual(chunks, ["0 1 2 3", "3 4 5 6"])
        self.assertEqual(lengths, [4, 4])
        self.assertEqual(chunk_text("", Tokenizer(), 4, 1), ([""], [0]))
        with self.assertRaises(ValueError):
            chunk_text("x", Tokenizer(), 4, 4)

    def test_diversity_and_undefined(self):
        self.assertEqual(topic_diversity([["a", "b"], ["b", "c"]], 2), 0.75)
        self.assertTrue(math.isnan(topic_diversity([])))
        self.assertIsNone(json_safe({"v": math.nan})["v"])

    def test_preprocessing_preserves_original(self):
        self.assertEqual(clean_text("The CAT! @Home", {"the"}), "cat! home")

    def test_cli_validation(self):
        p = parser()
        args = p.parse_args([])
        validate_args(args, p)
        self.assertEqual(args.topics, list(range(2, 25, 2)))
        self.assertIsNone(args.umap_seed)
        args.topics = [2, 2]
        with self.assertRaises(SystemExit):
            validate_args(args, p)


if __name__ == "__main__":
    unittest.main()
