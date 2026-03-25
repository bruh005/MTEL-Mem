import json
import pickle
import tempfile
import unittest
from pathlib import Path

from mtel_mem.core.rescore import aggregate_scores, load_query_records_jsonl, load_ranked_hits_jsonl, score_run
from mtel_mem.integrations.premem_locomo import export_premem_locomo


class PREMemLoCoMoExportTest(unittest.TestCase):
    def test_exporter_emits_mtel_mem_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            premem_results = root / "results.jsonl"
            qa_pkl = root / "qa.pkl"
            locomo_json = root / "locomo10.json"
            raw_session_pool = root / "raw_session_pool.pkl"
            lineage_json = root / "canonical_lineage.json"
            out_targets = root / "targets.jsonl"
            out_trace = root / "trace.jsonl"

            premem_results.write_text(
                json.dumps(
                    {
                        "question_id": "Q1",
                        "question_type": "single-hop",
                        "question": "What color is the bike?",
                        "answer": "Blue",
                        "retrieved_results": [
                            {"id": "reason_20", "score": 0.9},
                            {"id": "raw_10", "score": 0.8},
                            {"id": "raw_11", "score": 0.1},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with qa_pkl.open("wb") as handle:
                pickle.dump(
                    [
                        {
                            "question_id": "Q1",
                            "question_type": "single-hop",
                            "question": "What color is the bike?",
                            "answer": "Blue",
                            "session_pool": ["conv-1_session_1"],
                            "others": {},
                        }
                    ],
                    handle,
                )

            locomo_json.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "conv-1",
                            "conversation": {},
                            "event_summary": {},
                            "observation": {},
                            "session_summary": {},
                            "qa": [
                                {
                                    "question": "What color is the bike?",
                                    "answer": "Blue",
                                    "evidence": ["D1:3"],
                                    "category": 1,
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with raw_session_pool.open("wb") as handle:
                pickle.dump(
                    {
                        "Q1": [
                            {"id": 10, "session_id": "conv-1_session_1", "message_id": "D1:3"},
                            {"id": 11, "session_id": "conv-1_session_1", "message_id": "D1:4"},
                        ]
                    },
                    handle,
                )

            lineage_json.write_text(
                json.dumps(
                    {
                        "Q1": {
                            "reason_20": ["raw_10"],
                            "reason_21": ["raw_11"],
                        }
                    }
                ),
                encoding="utf-8",
            )

            summary = export_premem_locomo(
                premem_results_path=premem_results,
                qa_pickle_path=qa_pkl,
                locomo_json_path=locomo_json,
                raw_session_pool_path=raw_session_pool,
                out_targets_path=out_targets,
                out_trace_path=out_trace,
                canonical_lineage_json=lineage_json,
            )

            self.assertEqual(summary["queries"], 1)
            query_records = load_query_records_jsonl(out_targets)
            ranked_hits = load_ranked_hits_jsonl(out_trace)
            query_scores = score_run(query_records, ranked_hits, k=5)
            aggregate = aggregate_scores(query_scores)

            record = query_records["Q1"]
            self.assertEqual(record.targets.raw_ids, frozenset({"raw_10"}))
            self.assertEqual(record.targets.canonical_ids, frozenset({"reason_20"}))
            self.assertEqual(record.targets.source_ids, frozenset({"raw_10", "reason_20"}))
            self.assertGreater(aggregate["canonical"]["ndcg"], aggregate["raw"]["ndcg"])


if __name__ == "__main__":
    unittest.main()
