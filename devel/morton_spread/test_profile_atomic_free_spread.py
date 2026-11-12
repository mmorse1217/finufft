"""CPU-only parser/report tests: python3 -m unittest discover -s devel/morton_spread."""

import csv
from pathlib import Path
import tempfile
import unittest

import profile_atomic_free_spread as profile


DIAGNOSTICS = """prep upload_points=1 ms assign_bins=2 ms gpu_bin_sort=3 ms total=6 ms
spread warmup=2 timed=7 min=3 median=4 max=5 ms throughput=6 Mpoint/s 7 Mcell/s
ref method=3 setpts=8 ms execute=9 ms total=16 ms
"""
COMPARISONS = """comparison scope=spread warmup=2 timed=7 ours=4 ms method3=9 ms
comparison scope=gpu_pipeline warmup=2 timed=7 ours=10 ms method3=16 ms
comparison scope=host_pipeline warmup=2 timed=7 ours=12 ms method3=19 ms
"""


class TimingTests(unittest.TestCase):
    def test_gpu_scopes_and_independent_total(self):
        bench = profile.parse_bench_log(DIAGNOSTICS + COMPARISONS)
        for scope, ours, method3 in (("spread", 4, 9), ("gpu_pipeline", 10, 16),
                                     ("host_pipeline", 12, 19)):
            self.assertEqual(bench[f"comparison_{scope}_warmup"], 2)
            self.assertEqual(bench[f"comparison_{scope}_timed"], 7)
            self.assertEqual(bench[f"comparison_{scope}_ours_ms"], ours)
            self.assertEqual(bench[f"comparison_{scope}_method3_ms"], method3)
        self.assertEqual(bench["reference_total_ms"], 16)
        self.assertEqual(bench["prep_total_ms"], 6)
        self.assertEqual(bench["spread_median_ms"], 4)

    def test_cpu_and_legacy_missing_scopes(self):
        cpu = profile.parse_bench_log(COMPARISONS.replace(
            "comparison scope=gpu_pipeline warmup=2 timed=7 ours=10 ms method3=16 ms\n", ""))
        self.assertIsNone(cpu["comparison_gpu_pipeline_ours_ms"])
        self.assertEqual(cpu["comparison_host_pipeline_ours_ms"], 12)
        legacy = profile.parse_bench_log(DIAGNOSTICS +
            "timing setup=6 ms end_to_end=10 ms spread_vs_ref_exec=2.25x\n")
        self.assertIsNone(legacy["comparison_spread_ours_ms"])
        self.assertNotIn("end_to_end_ms", legacy)

    def test_report_and_csv_do_not_invent_comparisons(self):
        for text in (DIAGNOSTICS, DIAGNOSTICS + COMPARISONS):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                bench = profile.parse_bench_log(text)
                result = dict(bench=bench, key_metrics={}, args=[], workload="uniform",
                              variant="atomic_free", artifacts={})
                for key in ("stalls", "source_groups", "top_source_lines",
                            "top_sample_lines", "top_opcode_families"):
                    result[key] = []
                profile.write_summary(out, {}, {"uniform": result}, ["uniform"])
                report = (out / "summary.md").read_text()
                self.assertNotIn("atomic-free end-to-end", report)
                self.assertIn("not a sum of medians", report)
                self.assertIn("NCU replay differs", report)
                if text == DIAGNOSTICS:
                    self.assertIn("| uniform | host_pipeline | unavailable", report)
                else:
                    self.assertIn("| uniform | host_pipeline | 2 | 7 | 12.000 | 19.000 |", report)
                profile.write_csv(out / "bench.csv", list(bench), [bench])
                with (out / "bench.csv").open() as handle:
                    row = next(csv.DictReader(handle))
                self.assertEqual(row["comparison_host_pipeline_ours_ms"],
                                 "" if text == DIAGNOSTICS else "12.0")


if __name__ == "__main__":
    unittest.main()
