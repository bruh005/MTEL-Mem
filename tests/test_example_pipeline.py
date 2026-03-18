import unittest
from pathlib import Path

from mtel_mem.core.instability import compute_instability
from mtel_mem.core.rescore import aggregate_scores, load_query_records_jsonl, load_ranked_hits_jsonl, score_run


class ExamplePipelineTest(unittest.TestCase):
    def test_example_pipeline_runs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        query_records = load_query_records_jsonl(root / "examples" / "minimal_target_mappings.jsonl")
        ranked_hits = load_ranked_hits_jsonl(root / "examples" / "minimal_ranked_trace.jsonl")
        query_scores = score_run(query_records, ranked_hits, k=5)
        aggregate = aggregate_scores(query_scores)
        instability = compute_instability(query_scores, "raw", "canonical")

        self.assertEqual(set(query_scores.keys()), {"Q1", "Q2"})
        self.assertGreater(aggregate["source"]["ndcg"], aggregate["raw"]["ndcg"])
        self.assertGreater(instability["ndcg_changed"], 0.0)


if __name__ == "__main__":
    unittest.main()
