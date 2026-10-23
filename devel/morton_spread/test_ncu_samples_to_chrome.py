"""CPU-only regression tests: python3 -m unittest discover -s devel/morton_spread."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import ncu_samples_to_chrome as flamegraph


ROOT = "spread_cartesian_owned_block_shared"
HEADER = [
    "Line No", "Source", "Address", "Source",
    "Warp Stall Sampling (All Samples)",
    "Warp Stall Sampling (Not-issued Samples)",
    "Instructions Executed", "stall_short_sb", "stall_short_sb (Not Issued)",
]
SOURCE = """// A deliberately small source snapshot, not a CUDA compilation unit.
__device__ __forceinline__ void prepare_two_coefficients() {
  fold_rescale();
}
__device__ __forceinline__ void stage_block_quarter() {
  prepare_two_coefficients();
}
__device__ __forceinline__ void scan_one_staged_point() {
  consume_metadata();
}
__global__ void spread_cartesian_owned_block_shared() {
  stage_block_quarter();
  scan_one_staged_point<true>();
  __all_sync(0xffffffff, true);
}
"""


def sass(address, samples=5, inst=11):
    return ["", "", address, "IADD3 R1, R2, R3, RZ", samples, 2, inst, 3, 1]


def source_line(line, code):
    # CUDA aggregate metrics must not be counted alongside the SASS rows.
    return [line, code, "-", "-", 9999, 9999, 9999, 9999, 9999]


class FlamegraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / "exp_a_cartesian.cu"
        self.source.write_text(SOURCE)
        self.csv = self.directory / "capture.csv"

    def write_csv(self, rows):
        with self.csv.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerows([
                ["File Path", str(self.source)],
                ["Function Name", ROOT + "()"],
                HEADER,
                *rows,
            ])

    def parse(self):
        return flamegraph.parse(self.csv, ROOT)

    def tree(self, profile):
        layout = flamegraph.SourceLayout(self.source, ROOT)
        return flamegraph.build_tree(profile, layout)

    def test_duplicate_pcs_and_placeholders_are_not_counted_twice(self):
        self.write_csv([
            source_line(3, "fold_rescale();"), sass("0x10"),
            ["", "", "...", "...", 777, 777, 777, 777, 777],
            source_line(6, "prepare_two_coefficients();"), sass("0x10"),
            sass("0x20", samples=7, inst=13),
        ])
        profile = self.parse()
        self.assertEqual(profile.total_samples, 12)
        self.assertEqual(profile.total_instructions, 24)
        self.assertEqual(profile.unique_pcs, 2)
        self.assertEqual(profile.duplicate_pc_rows, 1)
        self.assertEqual(profile.lines[("exp_a_cartesian.cu", 3)]["src"], "fold_rescale();")
        self.assertEqual(profile.lines[("exp_a_cartesian.cu", 3)]["stalls"]["short_sb"], 3)
        self.assertEqual(profile.lines[("exp_a_cartesian.cu", 3)]["not_issued_stalls"]["short_sb"], 1)

    def test_pc_identity_is_scoped_to_selected_kernel(self):
        self.write_csv([
            ["Function Name", "spread_cartesian_owned()"],
            source_line(3, "fold_rescale();"), sass("0x10", samples=100, inst=100),
            ["Function Name", ROOT + "()"],
            source_line(3, "fold_rescale();"), sass("0x10"),
        ])
        profile = self.parse()
        self.assertEqual(profile.total_samples, 5)
        self.assertEqual(profile.total_instructions, 11)
        self.assertEqual(profile.duplicate_pc_rows, 0)

    def test_conflicting_duplicate_metrics_are_rejected(self):
        self.write_csv([
            source_line(3, "fold_rescale();"), sass("0x10"),
            source_line(6, "prepare_two_coefficients();"), sass("0x10", samples=6),
        ])
        with self.assertRaisesRegex(ValueError, "conflicting.*0x10"):
            self.parse()

    def test_missing_kernel_is_rejected(self):
        self.write_csv([["Function Name", "another_kernel()"], sass("0x10")])
        with self.assertRaisesRegex(ValueError, "no SASS.*" + ROOT):
            self.parse()

    def test_block_staging_includes_actual_coefficient_callee(self):
        self.write_csv([source_line(3, "fold_rescale();"), sass("0x10")])
        _, totals, _ = self.tree(self.parse())
        self.assertEqual(totals["prepare_two_coefficients"], 5)
        self.assertEqual(totals["stage_block_quarter"], 5)
        self.assertEqual(totals[ROOT], 5)

    def test_source_snapshot_controls_ranges_not_current_line_numbers(self):
        self.source.write_text("// shifted\n" * 40 + SOURCE)
        self.write_csv([source_line(43, "fold_rescale();"), sass("0x10")])
        _, totals, _ = self.tree(self.parse())
        self.assertEqual(totals["prepare_two_coefficients"], 5)

    def test_explicit_snapshot_can_have_a_different_filename(self):
        renamed = self.directory / "archived_kernel.cu"
        renamed.write_text(SOURCE)
        self.write_csv([source_line(3, "fold_rescale();"), sass("0x10")])
        layout = flamegraph.SourceLayout(renamed, ROOT)
        _, totals, _ = flamegraph.build_tree(self.parse(), layout)
        self.assertEqual(totals["prepare_two_coefficients"], 5)

    def test_old_coefficient_callee_is_discovered_without_a_version_map(self):
        self.source.write_text(SOURCE.replace("prepare_two_coefficients", "prepare_point_coefficients"))
        self.write_csv([source_line(3, "fold_rescale();"), sass("0x10")])
        _, totals, _ = self.tree(self.parse())
        self.assertEqual(totals["prepare_point_coefficients"], 5)
        self.assertEqual(totals["stage_block_quarter"], 5)

    def test_wrong_source_snapshot_fails_instead_of_misattributing(self):
        self.write_csv([source_line(3, "not_the_captured_source();"), sass("0x10")])
        with self.assertRaisesRegex(ValueError, "source.*mismatch"):
            self.tree(self.parse())

    def test_unmapped_samples_remain_visible_and_root_is_complete(self):
        self.write_csv([
            source_line(3, "fold_rescale();"), sass("0x10"),
            ["File Path", "/cuda/unknown.hpp"], ["Function Name", ROOT + "()"],
            HEADER, source_line(17, "unknown();"), sass("0x20", samples=7),
        ])
        profile = self.parse()
        _, totals, lines = self.tree(profile)
        self.assertEqual(totals[ROOT], profile.total_samples)
        self.assertEqual(totals["unmapped source"], 7)
        self.assertEqual(lines["unmapped source"][0][:2], ("unknown.hpp", 17))

    def test_vote_is_not_mislabeled_as_a_barrier(self):
        self.write_csv([
            ["File Path", "/cuda/sm_30_intrinsics.hpp"],
            ["Function Name", ROOT + "()"], HEADER,
            source_line(116, "return __nvvm_vote_all_sync(mask, pred);"), sass("0x10"),
        ])
        _, totals, _ = self.tree(self.parse())
        self.assertEqual(totals["__all_sync"], 5)
        self.assertEqual(totals.get("__syncwarp", 0), 0)

    def test_shared_inline_callee_is_not_counted_under_both_parents(self):
        self.source.write_text(SOURCE.replace(
            "  consume_metadata();", "  prepare_two_coefficients();"
        ))
        self.write_csv([source_line(3, "fold_rescale();"), sass("0x10")])
        layout = flamegraph.SourceLayout(self.source, ROOT)
        _, totals, _ = flamegraph.build_tree(self.parse(), layout)
        self.assertEqual(totals[ROOT], 5)
        self.assertIn("prepare_two_coefficients", layout.ambiguous_callers)

    def test_cli_outputs_complete_trace_with_explicit_sample_semantics(self):
        self.write_csv([
            source_line(3, "fold_rescale();"), sass("0x10"),
            source_line(6, "prepare_two_coefficients();"), sass("0x10"),
        ])
        output = self.directory / "trace.json"
        text = self.directory / "tree.txt"
        result = subprocess.run([
            sys.executable, flamegraph.__file__, str(self.csv), str(output),
            "--root", ROOT, "--kernel-us", "1234", "--text", str(text),
        ], capture_output=True, text=True, check=True)
        trace = json.loads(output.read_text())
        self.assertEqual(trace["otherData"]["total_samples"], 5)
        self.assertEqual(trace["otherData"]["duplicate_pc_rows"], 1)
        self.assertEqual(trace["otherData"]["total_warp_instructions"], 11)
        self.assertIn("not wall-clock", trace["otherData"]["note"])
        self.assertIn("100.0%", text.read_text().splitlines()[0])
        self.assertIn("5 samples", result.stdout)
        events = trace["traceEvents"]
        stack = []
        for event in events:
            if event["ph"] == "B":
                if stack:
                    self.assertGreaterEqual(event["ts"], stack[-1]["ts"])
                stack.append(event)
            elif event["ph"] == "E":
                start = stack.pop()
                self.assertGreaterEqual(event["ts"], start["ts"])
                if start["name"] == ROOT:
                    self.assertAlmostEqual(event["ts"], 1234)
        self.assertFalse(stack)

    def test_cli_rejects_nonfinite_or_nonpositive_scale(self):
        self.write_csv([source_line(3, "fold_rescale();"), sass("0x10")])
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value):
                result = subprocess.run([
                    sys.executable, flamegraph.__file__, str(self.csv),
                    str(self.directory / "trace.json"), "--root", ROOT,
                    "--kernel-us", value,
                ], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
