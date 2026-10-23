#!/usr/bin/env python3
"""Build a source-attribution flamegraph from an Nsight Compute CUDA/SASS CSV.

    ncu --import spread_ncu.ncu-rep --page source --print-source cuda,sass --csv \
        > spread_source.csv
    python3 ncu_samples_to_chrome.py spread_source.csv out.json \
        --kernel-us 12600 [--text out.txt] [--cuda-source exp_a_cartesian.cu]

PC sampling is flat: there is no device call stack or per-line elapsed time.
Each (kernel, SASS PC) is counted once, using its first source correlation.
The hierarchy is reconstructed from the matching prototype source snapshot;
shared inline callees with ambiguous parents are placed directly under the root.

Widths are sample counts. Chrome's time coordinates are only a synthetic layout
scaled to --kernel-us (the duration of the SAME profiled launch, not a benchmark
median). Neither the flamegraph nor its duration reference is a GPU timeline.
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys


@dataclass
class Profile:
    lines: dict
    unique_pcs: int
    duplicate_pc_rows: int
    kernel_name: str

    @property
    def total_samples(self):
        return sum(rec["samples"] for rec in self.lines.values())

    @property
    def total_instructions(self):
        return sum(rec["inst"] for rec in self.lines.values())


def parse(path, root="spread_cartesian_owned"):
    """Select one kernel and deduplicate PCs before aggregating source lines.

    Nsight repeats a PC's counters at inline definitions AND call sites. These
    are correlations, not independent executions. Keep the first correlation;
    reject conflicting counters rather than silently combining captures.
    CUDA aggregate rows and SASS '...' placeholders are never counted.
    """
    file = None
    line = None
    source = ""
    kernel = None
    columns = None
    row_width = 0
    seen = {}
    per_line = {}
    duplicates = 0

    def number(row, name):
        value = row[columns[name]].strip() if name in columns else ""
        return 0 if value in ("", "-") else int(value.replace(",", ""))

    with open(path, newline="") as input_file:
        for row in csv.reader(input_file):
            if not row:
                continue
            if row[0] == "File Path":
                file = Path(row[1]).name
                line, source = None, ""
                continue
            if row[0] == "Function Name":
                kernel = row[1]
                line, source = None, ""
                continue
            if row[0] == "Line No":
                columns = {name: i for i, name in enumerate(row)}
                row_width = len(row)
                continue
            if columns is None or len(row) != row_width:
                continue
            if kernel is None or kernel.split("(", 1)[0].strip() != root:
                continue
            if row[0].strip():
                line = None if row[0].strip() == "-" else int(row[0])
                source = row[1].strip()
            address = row[columns["Address"]].strip()
            if not re.fullmatch(r"0x[0-9a-fA-F]+", address):
                continue

            stalls = {}
            not_issued_stalls = {}
            for name in columns:
                if name.startswith("stall_"):
                    target = not_issued_stalls if " (Not Issued)" in name else stalls
                    target[name.removeprefix("stall_").removesuffix(" (Not Issued)")] = number(row, name)
            metrics = {
                "samples": number(row, "Warp Stall Sampling (All Samples)"),
                "not_issued_samples": number(row, "Warp Stall Sampling (Not-issued Samples)"),
                "inst": number(row, "Instructions Executed"),
                "thread_inst": number(row, "Thread Instructions Executed"),
                "predicated_thread_inst": number(row, "Predicated-On Thread Instructions Executed"),
                "stalls": stalls,
                "not_issued_stalls": not_issued_stalls,
            }
            key = (kernel, int(address, 16))
            if key in seen:
                if seen[key] != metrics:
                    raise ValueError(f"conflicting counters for {kernel} PC {address}; export one launch")
                duplicates += 1
                continue
            seen[key] = metrics
            rec = per_line.setdefault((file, line), {
                "src": source or row[columns["Source"]].strip(),
                "samples": 0, "not_issued_samples": 0, "inst": 0,
                "thread_inst": 0, "predicated_thread_inst": 0,
                "stalls": defaultdict(int), "not_issued_stalls": defaultdict(int),
            })
            for name, value in metrics.items():
                if isinstance(value, dict):
                    for reason, count in value.items():
                        rec[name][reason] += count
                else:
                    rec[name] += value

    kernels = {key[0] for key in seen}
    if not kernels:
        raise ValueError(f"no SASS rows for kernel {root}")
    if len(kernels) != 1:
        raise ValueError(f"multiple signatures for {root}; export one kernel")
    return Profile(per_line, len(seen), duplicates, kernels.pop())


def header_leaf(file, line, source):
    """Known inlined headers; unknown correlations remain explicitly unmapped."""
    if file == "spreadinterp.hpp":
        return "cudaFMA" if line == 47 else "fold_rescale"
    if file == "sm_32_intrinsics.hpp":
        return "__ldg"
    if file == "sm_30_intrinsics.hpp":
        if "vote_all_sync" in source:
            return "__all_sync"
        if "bar_warp_sync" in source:
            return "__syncwarp"
        if "__syncthreads" in source:
            return "__syncthreads"
    return "unmapped source"


class SourceLayout:
    """Source spans and static calls for this prototype's free CUDA functions.

    Read the archived .cu next to the CSV, not hard-coded line ranges from the
    latest kernel. This is intentionally not a general-purpose C++ parser.
    Strip comments/strings before matching definitions, braces and calls.
    """

    def __init__(self, path, root):
        self.path = Path(path)
        self.root = root
        text = self.path.read_text()
        self.source_lines = text.splitlines()
        ignored = r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
        code = re.sub(ignored, lambda m: re.sub(r"[^\n]", " ", m[0]), text)
        definition = re.compile(
            r"(?m)^(?:(?:__host__|__device__|__global__|__forceinline__)\s+)+"
            r"(?:void|int|float)\s+(\w+)\s*\("
        )
        self.ranges = []
        bodies = {}
        for match in definition.finditer(code):
            start = code.index("{", match.end())
            depth = 1
            end = start
            while depth and end + 1 < len(code):
                end += 1
                depth += (code[end] == "{") - (code[end] == "}")
            if depth:
                raise ValueError(f"unclosed CUDA function {match[1]} in {path}")
            self.ranges.append((code.count("\n", 0, match.start()) + 1,
                                code.count("\n", 0, end) + 1, match[1]))
            bodies[match[1]] = code[start + 1:end]
        if root not in bodies:
            raise ValueError(f"kernel {root} not found in source snapshot {path}")

        header_calls = {"fold_rescale", "cudaFMA", "__ldg", "__syncwarp",
                        "__all_sync", "__syncthreads"}
        names = set(bodies) | header_calls
        call = re.compile(r"\b(" + "|".join(sorted(names)) + r")\s*(?:<[^;{}()]*>)?\s*\(")
        graph = {name: list(dict.fromkeys(m[1] for m in call.finditer(body)))
                 for name, body in bodies.items()}
        graph["fold_rescale"] = ["cudaFMA"]
        self.reachable = set()

        def visit(frame, ancestors):
            if frame in ancestors:
                raise ValueError(f"recursive source call graph at {frame}")
            if frame not in self.reachable:
                self.reachable.add(frame)
                for child in graph.get(frame, []):
                    visit(child, ancestors | {frame})

        visit(root, set())
        self.children = {name: list(graph.get(name, [])) for name in self.reachable}
        parents = defaultdict(set)
        for parent, children in self.children.items():
            for child in children:
                parents[child].add(parent)
        # PC correlations cannot distinguish multiple inline callers. Do not
        # duplicate the shared callee's samples under every possible parent.
        self.ambiguous_callers = {
            child: sorted(callers) for child, callers in parents.items() if len(callers) > 1
        }
        for child, callers in sorted(self.ambiguous_callers.items()):
            for parent in callers:
                self.children[parent].remove(child)
            self.children[root].append(child)
        self.reachable.add("unmapped source")
        self.children[root].append("unmapped source")

    def leaf_of(self, file, line, source):
        if file in (self.path.name, "exp_a_cartesian.cu"):
            if line is not None:
                if not 1 <= line <= len(self.source_lines) or self.source_lines[line - 1].strip() != source:
                    raise ValueError(f"source snapshot mismatch at {file}:{line}; use --cuda-source from the capture")
                for lo, hi, name in self.ranges:
                    if lo <= line <= hi:
                        return name
            return "unmapped source"
        return header_leaf(file, line, source)


def build_tree(profile, layout):
    """Account for every deduplicated sample once, including unknown source."""
    self_samples = defaultdict(int)
    lines = defaultdict(list)
    for (file, line), rec in profile.lines.items():
        leaf = layout.leaf_of(file, line, rec["src"])
        if leaf not in layout.reachable:
            leaf = "unmapped source"
        self_samples[leaf] += rec["samples"]
        lines[leaf].append((file, line, rec))
    totals = {}

    def total(frame):
        totals[frame] = self_samples[frame] + sum(total(child) for child in layout.children.get(frame, []))
        return totals[frame]

    if total(layout.root) != profile.total_samples:
        raise ValueError("source tree does not account for all PC samples")
    for leaf in lines:
        lines[leaf].sort(key=lambda item: (item[0] or "", item[1] if item[1] is not None else -1))
    return self_samples, totals, lines


def positive_duration(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("duration must be finite and positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--kernel-us", type=positive_duration, required=True,
                        help="duration of the same ncu launch; used only to scale the sample layout")
    parser.add_argument("--text", type=Path)
    parser.add_argument("--root", default="spread_cartesian_owned")
    parser.add_argument("--cuda-source", type=Path,
                        help="matching source snapshot (default: exp_a_cartesian.cu next to CSV, then current source)")
    args = parser.parse_args()
    source = args.cuda_source
    if source is None:
        source = args.csv.with_name("exp_a_cartesian.cu")
        if not source.is_file():
            source = Path(__file__).with_name("exp_a_cartesian.cu")
    try:
        profile = parse(args.csv, args.root)
        layout = SourceLayout(source, args.root)
        self_samples, totals, lines = build_tree(profile, layout)
        if not profile.total_samples:
            raise ValueError("no PC samples for the selected kernel")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    total_samples = profile.total_samples
    us_per_sample = args.kernel_us / total_samples
    note = (
        "Flat ncu PC samples, deduplicated by (kernel, PC); first source correlation wins. "
        "Static source hierarchy, not a captured device call stack. Widths are samples, "
        "not wall-clock self time. Time coordinates are a synthetic sample layout scaled "
        "to the supplied kernel duration, not a GPU timeline. Short-scoreboard samples "
        "describe waiting consumers, not necessarily the source of the load."
    )
    events = [
        {"ph": "M", "name": "process_name", "pid": 1,
         "args": {"name": f"{args.root} (PC-sample attribution, NOT a timeline)"}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 1,
         "args": {"name": "synthetic source hierarchy; widths = samples"}},
        {"ph": "M", "name": "process_name", "pid": 2,
         "args": {"name": "Kernel duration reference (not a timeline)"}},
        {"ph": "M", "name": "thread_name", "pid": 2, "tid": 1,
         "args": {"name": "supplied duration of the same profiled launch"}},
    ]

    def begin(pid, offset, name, category, details):
        events.append({"ph": "B", "pid": pid, "tid": 1, "ts": offset * us_per_sample,
                       "name": name, "cat": category, "args": details})

    def end(pid, offset):
        events.append({"ph": "E", "pid": pid, "tid": 1, "ts": offset * us_per_sample})

    text = []

    def render(frame, offset, depth):
        width = totals[frame]
        if not width:
            return
        share = 100 * width / total_samples
        begin(1, offset, frame, "source attribution",
              {"samples": width, "sample_share_percent": share, "self_samples": self_samples[frame]})
        text.append(f"{'  ' * depth}{frame}  [{width:,} samples = {share:.1f}%]")
        cursor = offset
        for file, line, rec in lines[frame]:
            if not rec["samples"]:
                continue
            where = f"{file}:L{line}" if line is not None else (file or "unknown source")
            label = f"{where} {rec['src'][:100]}".rstrip()
            details = {
                "file": file, "line": line, "source": rec["src"],
                "samples": rec["samples"], "not_issued_samples": rec["not_issued_samples"],
                "warp_inst": rec["inst"], "thread_inst": rec["thread_inst"],
                "predicated_thread_inst": rec["predicated_thread_inst"],
                "stalls": {key: value for key, value in rec["stalls"].items() if value},
                "not_issued_stalls": {key: value for key, value in rec["not_issued_stalls"].items() if value},
            }
            begin(1, cursor, label, "source line", details)
            cursor += rec["samples"]
            end(1, cursor)
            text.append(f"{'  ' * (depth + 1)}{label}  "
                        f"[{rec['samples']:,} = {100 * rec['samples'] / total_samples:.1f}%]")
        for child in layout.children.get(frame, []):
            render(child, cursor, depth + 1)
            cursor += totals[child]
        if cursor != offset + width:
            raise ValueError(f"incomplete rendering of {frame}")
        end(1, cursor)

    render(args.root, 0, 0)
    begin(2, 0, f"{args.root} (duration reference)", "duration reference",
          {"duration_ms": args.kernel_us / 1000})
    end(2, total_samples)
    trace = {
        "traceEvents": events, "displayTimeUnit": "ms",
        "otherData": {
            "source": str(args.csv), "cuda_source": str(source),
            "kernel": profile.kernel_name, "total_samples": total_samples,
            "total_warp_instructions": profile.total_instructions,
            "unique_pcs": profile.unique_pcs, "duplicate_pc_rows": profile.duplicate_pc_rows,
            "unmapped_samples": totals["unmapped source"],
            "ambiguous_callers": layout.ambiguous_callers,
            "kernel_us": args.kernel_us, "note": note,
        },
    }
    args.out.write_text(json.dumps(trace, allow_nan=False) + "\n")
    text_output = "\n".join(text) + "\n\n" + note + "\n"
    if args.text:
        args.text.write_text(text_output)
    print(text_output, end="")
    print(f"\nwrote {args.out}: {total_samples:,} samples, {profile.total_instructions:,} warp instructions, "
          f"{profile.unique_pcs:,} PCs ({profile.duplicate_pc_rows} duplicate rows removed)")
    if totals["unmapped source"]:
        print(f"warning: {totals['unmapped source']:,} samples retained under 'unmapped source'", file=sys.stderr)
    if layout.ambiguous_callers:
        print("note: shared inline callees with ambiguous parents are attached to the root: "
              + ", ".join(sorted(layout.ambiguous_callers)), file=sys.stderr)


if __name__ == "__main__":
    main()
