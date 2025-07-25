import unittest
from benchmarks import (
    spelling_correctness,
    average_sentence_length,
    lexical_diversity,
    correctly_spelled_char_fraction,
    correctly_spelled_char_total,
    run_all,
)

class TestDatasetMetrics(unittest.TestCase):
    def test_metrics(self):
        text = "This is a simple sentence. Another one!"
        metrics = run_all(text)
        self.assertGreater(metrics['spelling_correctness'], 0.5)
        self.assertAlmostEqual(metrics['avg_sentence_length'], 3.0, delta=1.0)
        self.assertGreater(metrics['lexical_diversity'], 0.5)
        self.assertGreater(metrics['correct_char_fraction'], 0.5)
        self.assertEqual(
            metrics['correct_char_total'],
            correctly_spelled_char_total(text),
        )

    def test_color_codes(self):
        text = "This is a simple sentence. Another one!"
        colored = "\x1b[31mThis is a simple sentence.\x1b[0m Another one!"
        self.assertEqual(run_all(text), run_all(colored))

if __name__ == '__main__':
    unittest.main()
