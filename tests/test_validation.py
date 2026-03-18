import unittest

from mtel_mem.core.validation import (
    build_validation_report,
    run_null_control,
    run_paper_regression,
    run_positive_control,
    run_toy_case_suite,
    validate_ranked_hits,
)
from mtel_mem.schemas import RankedHit


class ValidationTest(unittest.TestCase):
    def test_toy_suite_is_exact(self) -> None:
        report = run_toy_case_suite()
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["failures"], [])

    def test_null_control_stays_quiet(self) -> None:
        report = run_null_control()
        self.assertEqual(report["false_positive_rate"], 0.0)

    def test_positive_control_detects_all_events(self) -> None:
        report = run_positive_control()
        self.assertEqual(report["detection_rate"], 1.0)

    def test_paper_regression_matches_expectations(self) -> None:
        report = run_paper_regression()
        self.assertTrue(report["all_scalars_within_tolerance"])
        self.assertTrue(report["all_density_checks_passed"])
        self.assertEqual(report["real_run_decision_change_count"], 5)

    def test_duplicate_ranks_fail_validation(self) -> None:
        ranked_hits = {
            "Q1": [
                RankedHit("x", "sys", "Q1", "a", 1),
                RankedHit("x", "sys", "Q1", "b", 1),
            ]
        }
        report = validate_ranked_hits(ranked_hits)
        self.assertEqual(report["failed_checks"], 1)

    def test_scorecard_metrics_are_positive(self) -> None:
        report = build_validation_report()
        self.assertEqual(report["engineering"]["schema_invariant_pass_rate"], 1.0)
        self.assertEqual(report["engineering"]["toy_case_metric_accuracy"], 1.0)
        self.assertEqual(report["scientific"]["null_control_false_positive_rate"], 0.0)
        self.assertEqual(report["scientific"]["positive_control_detection_rate"], 1.0)
        self.assertEqual(report["scientific"]["real_run_decision_change_count"], 5)


if __name__ == "__main__":
    unittest.main()
