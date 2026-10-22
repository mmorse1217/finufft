#!/usr/bin/env python3
"""Convert an Nsight Systems .sqlite export into a Chrome trace JSON.

Usage:
    nsys export --type=sqlite -o trace.sqlite trace.nsys-rep
    python3 nsys_to_chrome.py trace.sqlite trace_chrome.json

The trace has:
  * one GPU track per (device, stream) with every kernel / memset / memcpy,
    annotated with grid, block, registers and shared memory;
  * one row per CPU thread carrying the NVTX ranges (the prototype's phase
    markers and cufinufft/thrust ranges) and the CUDA runtime API calls.

Open the JSON in chrome://tracing or https://ui.perfetto.dev.
"""

import json
import re
import sqlite3
import sys
from collections import defaultdict


def short_name(name: str) -> str:
    if not name:
        return "<unknown>"
    n = name
    for needle, label in (
        ("spread_cartesian_owned", "spread_cartesian_owned"),
        ("spread_output_driven", "m3_spread_output_driven"),
        ("spread_3d_block_gather", "m4_spread_block_gather"),
        ("spread_nupts_driven", "m1_spread_nupts_driven"),
        ("calc_bin_size_noghost", "setpts_calc_bin_size"),
        ("calc_inverse_of_global_sort_idx", "setpts_inverse_sort_idx"),
        ("calc_subprob", "setpts_calc_subprob"),
        ("map_b_into_subprob", "setpts_map_b_into_subprob"),
    ):
        if needle in n:
            return label
    # generic: last qualified component before the first '('
    n = n.split("(")[0]
    n = n.split("::")[-1]
    n = re.sub(r"<.*", "", n)
    return n


def main(db_path: str, out_path: str) -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    strings = dict(cur.execute("select id, value from StringIds"))

    kernels = []
    for row in cur.execute(
        """select start, end, deviceId, streamId, demangledName, gridX, gridY, gridZ,
                  blockX, blockY, blockZ, registersPerThread, staticSharedMemory,
                  dynamicSharedMemory, localMemoryPerThread
           from CUPTI_ACTIVITY_KIND_KERNEL order by start"""
    ):
        kernels.append(row)
    memsets = list(
        cur.execute(
            "select start, end, deviceId, streamId, bytes from CUPTI_ACTIVITY_KIND_MEMSET"
        )
    )
    memcpys = list(
        cur.execute(
            "select start, end, deviceId, streamId, bytes, copyKind"
            " from CUPTI_ACTIVITY_KIND_MEMCPY"
        )
    )
    runtime = list(
        cur.execute(
            """select r.start, r.end, r.globalTid, s.value
               from CUPTI_ACTIVITY_KIND_RUNTIME r
               left join StringIds s on s.id = r.nameId order by r.start"""
        )
    )
    nvtx = list(
        cur.execute(
            """select n.start, n.end, n.text, n.textId, n.globalTid, n.domainId
               from NVTX_EVENTS n where n.end is not null order by n.start"""
        )
    )
    thread_names = list(
        cur.execute(
            """select t.globalTid, s.value from ThreadNames t
               left join StringIds s on s.id = t.nameId"""
        )
    )

    t0 = min([x[0] for x in kernels + memsets + memcpys + runtime] + [x[0] for x in nvtx])
    events = []
    summary = defaultdict(lambda: [0, 0.0, 0.0, 0.0])  # count, total, min, max

    def us(ns):
        return round((ns - t0) / 1000.0, 3)

    seen_tracks = set()

    def track(pid, tid, pname, tname):
        if (pid, tid) in seen_tracks:
            return
        seen_tracks.add((pid, tid))
        if pname is not None:
            events.append(
                {"ph": "M", "name": "process_name", "pid": pid, "args": {"name": pname}}
            )
        events.append(
            {"ph": "M", "name": "thread_name", "pid": pid, "tid": tid,
             "args": {"name": tname}}
        )

    for start, end, dev, stream, name_id, gx, gy, gz, bx, by, bz, regs, ssmem, dsmem, lmem in kernels:
        name = short_name(strings.get(name_id, ""))
        pid = 1000 + dev
        tid = 1000 + stream
        track(pid, tid, f"GPU {dev}", f"stream {stream}")
        dur = end - start
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": name,
                "pid": pid,
                "tid": tid,
                "ts": us(start),
                "dur": round(dur / 1000.0, 3),
                "args": {
                    "grid": f"{gx}x{gy}x{gz}",
                    "block": f"{bx}x{by}x{bz}",
                    "regs/thread": regs,
                    "static_smem": ssmem,
                    "dynamic_smem": dsmem,
                    "local_mem/thread": lmem,
                    "stream": stream,
                },
            }
        )
        s = summary[name]
        ms = dur / 1e6
        s[0] += 1
        s[1] += ms
        s[2] = min(s[2], ms) if s[2] else ms
        s[3] = max(s[3], ms)

    for start, end, dev, stream, nbytes in memsets:
        pid = 1000 + dev
        tid = 1000 + stream
        track(pid, tid, f"GPU {dev}", f"stream {stream}")
        events.append(
            {
                "ph": "X",
                "cat": "memset",
                "name": f"memset {nbytes} B",
                "pid": pid,
                "tid": tid,
                "ts": us(start),
                "dur": round((end - start) / 1000.0, 3),
                "args": {"bytes": nbytes},
            }
        )

    for start, end, dev, stream, nbytes, kind in memcpys:
        pid = 1000 + dev
        tid = 1000 + stream
        track(pid, tid, f"GPU {dev}", f"stream {stream}")
        events.append(
            {
                "ph": "X",
                "cat": "memcpy",
                "name": f"memcpy {nbytes} B (kind {kind})",
                "pid": pid,
                "tid": tid,
                "ts": us(start),
                "dur": round((end - start) / 1000.0, 3),
                "args": {"bytes": nbytes, "copyKind": kind},
            }
        )

    for gid, name in thread_names:
        pid, tid = gid >> 32, gid & 0xFFFFFFFF
        track(pid, tid, None, name)

    for start, end, text, text_id, gid, domain in nvtx:
        name = text if text else strings.get(text_id, "")
        if not name:
            continue
        pid, tid = gid >> 32, gid & 0xFFFFFFFF
        track(pid, tid, None, "main" if tid == (gid & 0xFFFFFFFF) else f"thread {tid}")
        events.append(
            {
                "ph": "X",
                "cat": "nvtx" if domain == 0 else f"nvtx-domain-{domain}",
                "name": name,
                "pid": pid,
                "tid": tid,
                "ts": us(start),
                "dur": round((end - start) / 1000.0, 3),
            }
        )

    for start, end, gid, name in runtime:
        if not name:
            continue
        pid, tid = gid >> 32, gid & 0xFFFFFFFF
        track(pid, tid, None, "main" if tid == (gid & 0xFFFFFFFF) else f"thread {tid}")
        events.append(
            {
                "ph": "X",
                "cat": "cuda_api",
                "name": name,
                "pid": pid,
                "tid": tid,
                "ts": us(start),
                "dur": round((end - start) / 1000.0, 3),
            }
        )

    events.sort(key=lambda e: (e.get("ts", 0.0), e.get("pid", 0), e.get("tid", 0)))
    trace = {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "otherData": {
            "source": db_path,
            "tool": "nsys_to_chrome.py",
        },
    }
    with open(out_path, "w") as f:
        json.dump(trace, f)

    print(f"wrote {out_path}: {len(events)} events")
    print()
    print(f"{'kernel':34s} {'launches':>8s} {'total ms':>10s} {'min ms':>9s} {'max ms':>9s}")
    for name, (count, total, mn, mx) in sorted(summary.items(), key=lambda kv: -kv[1][1]):
        print(f"{name:34s} {count:8d} {total:10.3f} {mn:9.3f} {mx:9.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
