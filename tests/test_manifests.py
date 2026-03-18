import unittest

from mtel_mem.adapters.base import load_manifest, validate_manifest
from mtel_mem.adapters.locomo import default_manifest_path as locomo_manifest_path
from mtel_mem.adapters.longmemeval_s import default_manifest_path as longmemeval_s_manifest_path


class ManifestTest(unittest.TestCase):
    def test_locomo_manifest_is_valid(self) -> None:
        summary = validate_manifest(load_manifest(locomo_manifest_path()))
        self.assertEqual(summary["missing_files"], [])
        self.assertEqual(summary["missing_trace_examples"], [])
        self.assertTrue(summary["counts_match"])

    def test_longmemeval_manifest_is_valid(self) -> None:
        summary = validate_manifest(load_manifest(longmemeval_s_manifest_path()))
        self.assertEqual(summary["missing_files"], [])
        self.assertEqual(summary["missing_trace_examples"], [])
        self.assertTrue(summary["counts_match"])


if __name__ == "__main__":
    unittest.main()
