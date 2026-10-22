#!/usr/bin/env python3
"""Build a nested Chrome trace from an Nsight Compute source-page CSV.

    ncu --import spread_ncu.ncu-rep --page source --print-source cuda,sass --csv \
        > spread_ncu_source_cuda_sass.csv
    python3 ncu_samples_to_chrome.py spread_ncu_source_cuda_sass.csv out.json \
        --kernel-us 21650

The trace is a three/four-level stack that mirrors the real (inlined) call structure:

    spread_cartesian_owned                     kernel
      stage_raw                                inlined device function
        <source line>
      stage_coeff                              inlined device function
        fold_rescale / cudaFMA                 helper inlined into stage_coeff
          <source line>
        <source line>
      dim_support                              inlined device function
        <source line>
      kernel body: <phase>                     inline code, grouped (not a function)
        <source line>

Widths are PC-sampling samples rescaled so the root is the measured kernel duration. It
is an attribution view (children partition their parent's samples), not a wall-clock
timeline; the stack levels *are* the real call structure, but the widths are samples, not
self time. `kernel body: ...` groups are inline code, labelled as such.
"""

import csv
import json
import sys
from collections import defaultdict

# Function ranges in devel/morton_spread/exp_a_cartesian.cu (current kernel).
# Function ranges in devel/morton_spread/exp_a_cartesian.cu (current kernel).
FUNC_RANGES = [
    (198, 214, "stage_raw"),
    (220, 236, "stage_coeff"),
    (241, 255, "dim_support"),
    (259, 297, "scan_point"),
    (328, 335, "kernel body: coefficient preload"),
    (341, 352, "kernel body: bin loop"),
    (353, 359, "kernel body: batch staging"),
    (360, 365, "kernel body: scan calls"),
    (366, 366, "kernel body: syncwarp"),
    (372, 383, "kernel body: final store"),
]

# Header files: (function they are inlined into, helper name)
HEADER_FUNCS = {
    "spreadinterp.hpp": ("stage_coeff", "fold_rescale"),
    "sm_30_intrinsics.hpp": ("kernel body: syncwarp", "__syncwarp"),
    "sm_32_intrinsics.hpp": ("stage_raw", "__ldg"),
}

# Source-line labels (fall back to the source text).
SHORT_LABEL = {
    198: "stage_raw()",
    207: "load x/y/z/strength",
    220: "stage_coeff()",
    227: "fold_rescale",
    228: "support start ceil",
    229: "zz rescale",
    230: "write support start",
    233: "8x Horner (register coeffs)",
    234: "write kernel coefficient",
    241: "dim_support()",
    244: "wrap d<0",
    245: "wrap d>=nf",
    249: "low test",
    252: "n select",
    253: "j0 select",
    254: "l0 select",
    259: "scan_point()",
    262: "load s (float4) + strength",
    269: "early-out",
    282: "wxy product",
    287: "weight product",
    288: "acc x",
    293: "acc y",
    328: "coeff preload",
    332: "load 9 coefficients",
    341: "bin loop ix",
    344: "bin loop iy",
    347: "bin loop iz",
    349: "bin index",
    350: "binstart loads",
    353: "batch loop (split stride)",
    355: "stage_raw call",
    357: "coeff pass",
    358: "stage_coeff call",
    360: "scan call loop",
    362: "scan_point(b)",
    363: "scan_point(b+1)",
    365: "scan_point(tail)",
    366: "syncwarp",
    376: "final store loop",
    379: "store guard",
    381: "store",
}


def parse(path):
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


def func_of(file, line):
    if file == "exp_a_cartesian.cu":
        for lo, hi, name in FUNC_RANGES:
            if line is not None and lo <= line <= hi:
                return name, None
        return "kernel body: other", None
    if file in HEADER_FUNCS:
        func, helper = HEADER_FUNCS[file]
        return func, helper
    return "other", None


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    src, out = sys.argv[1], sys.argv[2]
    kernel_us = 21650.0
    if "--kernel-us" in sys.argv:
        kernel_us = float(sys.argv[sys.argv.index("--kernel-us") + 1])

    per_line = parse(src)
    total = sum(v["samples"] for v in per_line.values())
    if not total:
        sys.exit("no samples in the source CSV")
    us_per_sample = kernel_us / total

    # func -> helper -> [(line, rec)]
    tree = defaultdict(lambda: defaultdict(list))
    for (file, line), rec in per_line.items():
        func, helper = func_of(file, line)
        tree[func][helper].append((file, line, rec))

    events = [
        {"ph": "M", "name": "process_name", "pid": 1,
         "args": {"name": "spread_cartesian_owned (PC-sample attribution, call tree)"}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 1,
         "args": {"name": "kernel -> inlined function -> helper -> source line"}},
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


    begin(1, 1, 0, "spread_cartesian_owned", "kernel", {"samples": total,
                                                       "measured_ms": kernel_us / 1000.0})
    offset = 0
    # Keep a stable, readable order: real functions first, then inline groups.
    order = ["stage_raw", "stage_coeff", "dim_support"] + \
            sorted(f for f in tree if f.startswith("kernel body")) + \
            [f for f in tree if f not in ("stage_raw", "stage_coeff", "dim_support")
             and not f.startswith("kernel body")]
    for func in order:
        if func not in tree:
            continue
        func_samples = sum(rec["samples"] for entries in tree[func].values()
                           for _, _, rec in entries)
        begin(1, 1, offset, func, "function" if not func.startswith("kernel body")
              else "inline", {"samples": func_samples})
        foff = offset
        # Real helper frames first (they are callees), then this function's own lines.
        helpers = sorted(h for h in tree[func] if h is not None)
        for helper in helpers:
            entries = sorted(tree[func][helper], key=lambda e: (e[0], e[1] if e[1] is not None else -1))
            hs = sum(rec["samples"] for _, _, rec in entries)
            begin(1, 1, foff, helper, "helper", {"samples": hs})
            hoff = foff
            for file, line, rec in entries:
                label = SHORT_LABEL.get(line, rec["src"][:60])
                begin(1, 1, hoff, f"{file}:L{line} {label}", "line",
                      {"file": file, "line": line, "samples": rec["samples"],
                       "warp_inst": rec["inst"],
                       "stalls": {k: v for k, v in rec["stalls"].items() if v}})
                hoff += rec["samples"]
                end(1, 1, hoff)
            end(1, 1, foff + hs)
            foff += hs
        entries = sorted(tree[func].get(None, []),
                         key=lambda e: (e[0], e[1] if e[1] is not None else -1))
        for file, line, rec in entries:
            label = SHORT_LABEL.get(line, rec["src"][:60])
            begin(1, 1, foff, f"L{line} {label}", "line",
                  {"file": file, "line": line, "samples": rec["samples"],
                   "warp_inst": rec["inst"],
                   "stalls": {k: v for k, v in rec["stalls"].items() if v}})
            foff += rec["samples"]
            end(1, 1, foff)
        end(1, 1, offset + func_samples)
        offset += func_samples
    end(1, 1, total)

    begin(2, 1, 0, "spread_cartesian_owned (measured)", "kernel",
          {"duration_ms": kernel_us / 1000.0})
    end(2, 1, total)

    trace = {"traceEvents": events, "displayTimeUnit": "ms",
             "otherData": {"source": src, "total_samples": total, "kernel_us": kernel_us,
                           "note": "PC-sample attribution over the real inlined call tree; "
                                   "width is samples, not self time"}}
    json.dump(trace, open(out, "w"))
    print(f"wrote {out}: {len(events)} events, {total:,} samples, root {kernel_us/1000:.2f} ms")
    print(f"{'frame':40s} {'samples':>9s} {'share':>7s}")
    for func in order:
        if func not in tree:
            continue
        s = sum(rec["samples"] for entries in tree[func].values() for _, _, rec in entries)
        print(f"{func:40s} {s:9,d} {100*s/total:6.1f}%")


if __name__ == "__main__":
    main()
