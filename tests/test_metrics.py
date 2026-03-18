import math
import unittest

from mtel_mem.core.metrics import mrr_at_k, ndcg_at_k, recall_at_k


class MetricsTest(unittest.TestCase):
    def test_metrics_return_expected_values(self) -> None:
        ranked_ids = ["canon_47", "turn_47", "other"]
        target_ids = {"turn_47"}

        self.assertEqual(recall_at_k(ranked_ids, target_ids, 3), 1.0)
        self.assertEqual(mrr_at_k(ranked_ids, target_ids, 3), 0.5)
        self.assertTrue(math.isclose(ndcg_at_k(ranked_ids, target_ids, 3), 1.0 / math.log2(3), rel_tol=1e-9))

    def test_nan_for_empty_targets(self) -> None:
        ranked_ids = ["x", "y"]
        self.assertTrue(math.isnan(recall_at_k(ranked_ids, set(), 2)))
        self.assertTrue(math.isnan(mrr_at_k(ranked_ids, set(), 2)))
        self.assertTrue(math.isnan(ndcg_at_k(ranked_ids, set(), 2)))


if __name__ == "__main__":
    unittest.main()
