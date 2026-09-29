#!/usr/bin/env python3
"""Hardware-independent checks that the acceptance oracle cannot hide failures."""
import tempfile
import unittest
from pathlib import Path

from acceptance import Acceptance, compare, rows
from fixtures import generate


class AcceptanceTests(unittest.TestCase):
    def test_fixtures_are_deterministic_and_cross_both_thresholds(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a, b = generate(first), generate(second)
            self.assertEqual(a, b)
            for data in a["datasets"].values():
                threshold = data["threshold"]
                self.assertEqual(data["subject_lengths"], [threshold - 1, threshold, threshold + 1, threshold + 3])

    def test_comparison_preserves_multiplicity_coordinates_and_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left", Path(directory) / "right"
            row = "q\ts\t1\t10\t1\t10\t100\t10\t10\t0\t0\t0\t20\t10\t1e-9\t1\t1\n"
            second = row.replace("q\ts", "other\ts")
            left.write_text(row + second)
            right.write_text(second + row.replace("1e-9", "0.000000001"))
            compare(left, right)
            for bad in (row, row + second + second, row.replace("\t20\t", "\t21\t") + second,
                        row.replace("\t1\t10", "\t2\t10", 1) + second):
                right.write_text(bad)
                with self.assertRaises(AssertionError):
                    compare(left, right)
            right.write_text("diagnostic on stdout\n")
            with self.assertRaises(AssertionError):
                rows(right)

    def test_discovery_does_not_count_as_kernel_evidence(self):
        trace = 'GBLASTN_DEVICE backend=cuda id=0 name="NVIDIA Thor" compute_capability=11.0\n'
        trace += "GBLASTN_KERNEL branch=small phase=scan status=completed\n"
        trace += "GBLASTN_KERNEL branch=small phase=extend status=completed\n"
        for length in (499999, 500000, 500001, 500003):
            trace += f"GBLASTN_DISPATCH branch=small subject_length={length} backend={'cuda' if length > 500000 else 'cpu'}\n"
        Acceptance.evidence(trace, "small")
        for bad in ("NVIDIA Thor cc=11.0", trace.replace("phase=extend", "phase=allocation"),
                    trace.replace("500000 backend=cpu", "500000 backend=cuda")):
            with self.assertRaises(AssertionError):
                Acceptance.evidence(bad, "small")


if __name__ == "__main__":
    unittest.main()
