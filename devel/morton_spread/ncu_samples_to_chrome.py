#!/usr/bin/env python3
"""Build a nested Chrome trace from an Nsight Compute source-page CSV.

    ncu --import spread_ncu.ncu-rep --page source --print-source cuda,sass --csv \
        > spread_ncu_source_cuda_sass.csv
    python3 ncu_samples_to_chrome.py spread_ncu_source_cuda_sass.csv out.json \
        --kernel-us 16590 [--text out.txt]

Important: Nsight Compute PC sampling is *flat*. It attributes samples to a SASS
instruction and, with `-lineinfo`, to a source line. It does not record a device call
stack, and the kernel's device functions are inlined, so there is no runtime stack to
recover. `.nsys-rep` is a timeline (kernel/API events), not a call-stack trace either.

What this script does is apply the **static call graph from the source** to the flat
sample attribution. The resulting hierarchy is:

    spread_cartesian_owned                     kernel
      kernel body: batch loop + scan calls     inline code
        stage_raw_points                       inlined device function
          __ldg                                intrinsic inlined into stage_raw_points
        prepare_point_coefficients             inlined device function
          fold_rescale                         helper inlined into prepare_point_coefficients
            cudaFMA                            helper inlined into fold_rescale
        scan_one_staged_point                  inlined device function
          place_support                        helper inlined into scan_one_staged_point
        __syncwarp                             intrinsic inlined into the batch loop
      kernel body: x-run bin loop              inline code
      kernel body: final store                 inline code

A parent's width is its own samples plus all descendants' samples, so `scan_point`
includes `dim_support`. This is an attribution view, not a wall-clock timeline; the
stack levels are the source call structure, while widths are PC samples scaled to the
measured kernel duration. `kernel body: ...` frames are inline code, not functions.
"""

import csv
import json
import sys
from collections import defaultdict

# Leaf frames in devel/morton_spread/exp_a_cartesian.cu (current v11ay kernel).
# These are the source ranges whose samples belong to that function/group.
FUNC_RANGES = [
    (262, 279, "stage_raw_points"),
    (287, 307, "prepare_point_coefficients"),
    (312, 342, "prepare_two_coefficients"),
    (347, 361, "place_support"),
    (366, 374, "place_support_nowrap"),
    (379, 431, "scan_one_staged_point"),
    # Per-warp kernel body (spread_cartesian_owned).
    (460, 516, "kernel body: coefficient preload"),
    (517, 539, "kernel body: x-run bin loop"),
    (540, 615, "kernel body: batch loop + scan calls"),
    (616, 634, "kernel body: final store"),
    # Block-shared clustered kernel body (spread_cartesian_owned_block_shared).
    (646, 673, "stage_block_quarter"),
    (690, 769, "kernel body: block setup/preload"),
    (770, 802, "kernel body: block x-run bin loop"),
    (803, 901, "kernel body: block batch loop + scan calls"),
    (902, 923, "kernel body: block final store"),
]

# Static source call graph. The values are children in source/call order. Frames
# not listed as children of anything are top-level kernel-body groups.
CHILDREN = {
    "spread_cartesian_owned": [
        "kernel body: coefficient preload",
        "kernel body: x-run bin loop",
        "kernel body: batch loop + scan calls",
        "kernel body: final store",
        "kernel body: other",
    ],
    "spread_cartesian_owned_block_shared": [
        "kernel body: block setup/preload",
        "kernel body: block x-run bin loop",
        "kernel body: block batch loop + scan calls",
        "kernel body: block final store",
        "kernel body: other",
    ],
    "kernel body: batch loop + scan calls": [
        "stage_raw_points",
        "prepare_two_coefficients",
        "scan_one_staged_point",
        "__syncwarp",
    ],
    "kernel body: block batch loop + scan calls": [
        "stage_block_quarter",
        "scan_one_staged_point",
        "__syncthreads",
    ],
    "stage_block_quarter": [
        "stage_raw_points",
        "prepare_two_coefficients",
        "__syncwarp",
    ],
    "stage_raw_points": ["__ldg"],
    "prepare_point_coefficients": ["fold_rescale"],
    "prepare_two_coefficients": ["fold_rescale"],
    "fold_rescale": ["cudaFMA"],
    "scan_one_staged_point": ["place_support", "place_support_nowrap"],
}

# Header-file leaves. These files are inlined into the kernel; the CSV has no
# inlined call stack, only the header source line (sometimes no line at all).
def header_leaf(file, line, src=""):
    if file == "spreadinterp.hpp":
        # cudaFMA body is one line; fold_rescale is the surrounding helper.
        if line == 47:
            return "cudaFMA"
        return "fold_rescale"
    if file == "sm_30_intrinsics.hpp":
        if "__syncthreads" in src:
            return "__syncthreads"
        return "__syncwarp"
    if file == "sm_32_intrinsics.hpp":
        return "__ldg"
    return "kernel body: other"


# Source-line labels (fall back to the source text).
SHORT_LABEL = {
    266: "stage x (ldg + shared store)",
    267: "stage y (shared store)",
    268: "stage z (shared store)",
    269: "load strength (float2)",
    270: "stage strength re",
    271: "stage strength im",
    318: "fold_rescale",
    320: "support start ceil (float)",
    322: "support start int",
    323: "horner argument",
    326: "write support start",
    328: "8x Horner (index k)",
    333: "write coefficient k",
    335: "8x Horner (index k+3, shared)",
    341: "write coefficient k+3",
    347: "relative support start",
    348: "wrap relative start",
    353: "below_group test",
    354: "wraps_high test",
    355: "wrapped cell count",
    356: "group cell count select",
    357: "coefficient offset select",
    358: "local cell offset select",
    368: "intersection begin",
    370: "intersection end",
    371: "group cell count",
    372: "coefficient offset",
    373: "local cell offset",
    382: "metadata LDS (int2 + z + strength)",
    388: "support placement x",
    389: "support placement y",
    390: "support placement z",
    396: "early-out",
    399: "x coefficient load",
    400: "x lane-group shift select",
    403: "y coefficient base",
    404: "y coefficient load (contiguous)",
    409: "z coefficient base",
    410: "z coefficient load (contiguous)",
    413: "xy weight products",
    416: "strength*z real/imag (z0)",
    417: "strength*z real/imag (z1)",
    418: "strength*z real/imag (z2)",
    419: "strength*z real/imag (z3)",
    428: "cell accumulation x",
    429: "cell accumulation y",
    656: "quarter begin",
    657: "quarter end",
    660: "stage raw quarter",
    669: "prepare coefficients (3-point)",
    672: "quarter syncwarp",
    756: "initial block sync",
    809: "stage current batch",
    815: "publish current batch",
    824: "stage next batch",
    890: "publish next batch / end scan",
}


def parse(path):
    """Return {(file, line): {src, samples, inst, stalls}} from an ncu source CSV."""
    rows = list(csv.reader(open(path, newline="")))
    file = None
    line = None
    hdr = None
    idx = None
    per = {}
    for r in rows:
        if not r:
            continue
        if r[0] == "File Path":
            file = r[1].split("/")[-1]
            line = None
            continue
        if r[0] == "Line No":
            hdr = r
            idx = {h: i for i, h in enumerate(hdr)}
            continue
        if r[0] == "Function Name" or hdr is None or len(r) != len(hdr):
            continue
        v = r[0].strip()
        if v and v != "-":
            try:
                line = int(v)
            except ValueError:
                continue
        if r[idx["Address"]] == "-":
            continue
        try:
            samples = int(r[idx["Warp Stall Sampling (All Samples)"]] or 0)
        except ValueError:
            samples = 0
        try:
            inst = int(r[idx["Instructions Executed"]] or 0)
        except ValueError:
            inst = 0
        rec = per.setdefault((file, line), {"src": r[1].strip(), "samples": 0, "inst": 0,
                                            "stalls": defaultdict(int)})
        rec["samples"] += samples
        rec["inst"] += inst
        for k, i in idx.items():
            if k.startswith("stall_") and "Not Issued" not in k:
                try:
                    rec["stalls"][k.replace("stall_", "")] += int(r[i] or 0)
                except ValueError:
                    pass
    return per


def leaf_of(file, line, src=""):
    """Classify one source line into the leaf frame it belongs to."""
    if file == "exp_a_cartesian.cu":
        for lo, hi, name in FUNC_RANGES:
            if line is not None and lo <= line <= hi:
                return name
        return "kernel body: other"
    return header_leaf(file, line, src)


def build_tree(per_line, root):
    """Return (self_samples, total_samples, lines) for one kernel root.

    `lines[leaf]` holds the source rows that belong directly to that frame;
    samples from descendants are aggregated into `total_samples` through the
    static call graph rooted at `root`, never duplicated into `self_samples`.
    Shared leaf names (stage_raw_points, prepare_point_coefficients, ...) are
    attached to different parents depending on the selected kernel.
    """
    reachable = set()
    def visit(frame):
        if frame in reachable:
            return
        reachable.add(frame)
        for kid in CHILDREN.get(frame, []):
            visit(kid)
    visit(root)

    self_samples = defaultdict(int)
    lines = defaultdict(list)
    for (file, line), rec in per_line.items():
        leaf = leaf_of(file, line, rec.get("src", ""))
        if leaf not in reachable:
            continue
        self_samples[leaf] += rec["samples"]
        lines[leaf].append((file, line, rec))

    # Include zero-sample frames so the hierarchy stays visible.
    for frame in reachable:
        self_samples.setdefault(frame, 0)

    total = dict(self_samples)
    changed = True
    while changed:  # Tree is shallow; simple fixed-point aggregation.
        changed = False
        for parent in reachable:
            kids = CHILDREN.get(parent, [])
            candidate = self_samples[parent] + sum(total.get(k, 0) for k in kids
                                                    if k in reachable)
            if candidate != total.get(parent, 0):
                total[parent] = candidate
                changed = True

    for leaf in lines:
        lines[leaf].sort(key=lambda e: (e[0], e[1] if e[1] is not None else -1))
    return self_samples, total, lines


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    src, out = sys.argv[1], sys.argv[2]
    kernel_us = 21650.0
    if "--kernel-us" in sys.argv:
        kernel_us = float(sys.argv[sys.argv.index("--kernel-us") + 1])
    text_out = None
    if "--text" in sys.argv:
        text_out = sys.argv[sys.argv.index("--text") + 1]
    root = "spread_cartesian_owned"
    if "--root" in sys.argv:
        root = sys.argv[sys.argv.index("--root") + 1]

    per_line = parse(src)
    total_samples = sum(v["samples"] for v in per_line.values())
    if not total_samples:
        sys.exit("no samples in the source CSV")
    us_per_sample = kernel_us / total_samples

    self_samples, total, lines = build_tree(per_line, root)

    events = [
        {"ph": "M", "name": "process_name", "pid": 1,
         "args": {"name": f"{root} (PC-sample attribution, source call tree)"}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 1,
         "args": {"name": "kernel -> source call hierarchy -> source line"}},
        {"ph": "M", "name": "process_name", "pid": 2,
         "args": {"name": "GPU timeline (measured)"}},
        {"ph": "M", "name": "thread_name", "pid": 2, "tid": 1,
         "args": {"name": "one ncu launch"}},
    ]

    def T(offset):
        return offset * us_per_sample

    def begin(pid, tid, a, name, cat, args=None):
        e = {"ph": "B", "pid": pid, "tid": tid, "ts": T(a), "name": name, "cat": cat}
        if args:
            e["args"] = args
        events.append(e)

    def end(pid, tid, b):
        events.append({"ph": "E", "pid": pid, "tid": tid, "ts": T(b)})

    def line_args(file, line, rec):
        return {"file": file, "line": line, "samples": rec["samples"],
                "warp_inst": rec["inst"],
                "stalls": {k: v for k, v in rec["stalls"].items() if v}}

    def render(frame, offset, depth):
        """Emit a frame, its own source lines, then its callees."""
        width = total.get(frame, 0)
        if width == 0:
            return
        cat = "function" if not frame.startswith("kernel body") else "inline"
        begin(1, 1, offset, frame, cat,
              {"samples": width, "self_samples": self_samples.get(frame, 0)})
        cursor = offset
        # Own source lines first (source order), then nested callees.
        for file, line, rec in lines.get(frame, []):
            label = SHORT_LABEL.get(line, rec["src"][:60])
            name = f"{file}:L{line} {label}" if line is not None else f"{file} {label}"
            begin(1, 1, cursor, name, "line", line_args(file, line, rec))
            cursor += rec["samples"]
            end(1, 1, cursor)
        for kid in CHILDREN.get(frame, []):
            render(kid, cursor, depth + 1)
            cursor += total.get(kid, 0)
        end(1, 1, offset + width)

    render(root, 0, 0)

    begin(2, 1, 0, f"{root} (measured)", "kernel",
          {"duration_ms": kernel_us / 1000.0})
    end(2, 1, total_samples)

    trace = {"traceEvents": events, "displayTimeUnit": "ms",
             "otherData": {
                 "source": src,
                 "total_samples": total_samples,
                 "kernel_us": kernel_us,
                 "note": ("Flat ncu PC samples attributed through the static source call "
                          "graph; parent width = own samples + descendants. Widths are "
                          "samples, not wall-clock self time. Inlined functions have no "
                          "runtime call stack."),
             }}
    json.dump(trace, open(out, "w"))

    # Text rendering of the same hierarchy (handy for review and diffs).
    text = []

    def text_render(frame, depth):
        width = total.get(frame, 0)
        if width == 0:
            return
        share = 100.0 * width / total_samples
        text.append(f"{'  ' * depth}{frame}  [{width:,} samples = {share:.1f}%]")
        for file, line, rec in lines.get(frame, []):
            label = SHORT_LABEL.get(line, rec["src"][:60])
            where = f"{file}:L{line}" if line is not None else file
            lshare = 100.0 * rec["samples"] / total_samples
            text.append(f"{'  ' * (depth + 1)}{where} {label}  "
                        f"[{rec['samples']:,} = {lshare:.1f}%]")
        for kid in CHILDREN.get(frame, []):
            text_render(kid, depth + 1)

    text_render(root, 0)
    text = "\n".join(text) + "\n"
    if text_out:
        open(text_out, "w").write(text)
    print(text, end="")

    print(f"\nwrote {out}: {len(events)} events, {total_samples:,} samples, "
          f"root {kernel_us/1000:.2f} ms")
    print(f"note: hierarchy is the source call graph; PC samples are flat.")


if __name__ == "__main__":
    main()
