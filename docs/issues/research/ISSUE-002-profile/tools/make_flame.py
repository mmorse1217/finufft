#!/usr/bin/env python3
import csv, sqlite3, sys, os
from xml.sax.saxutils import escape

def read_nvtx(path):
    rows = []
    with open(path, newline="") as f:
        for line in f:
            if not line or line[0] not in "0123456789":
                continue
            parts = next(csv.reader([line]))
            rows.append({
                "start": int(parts[0]), "end": int(parts[1]),
                "dur": int(parts[2]), "name": parts[5],
                "lvl": int(parts[8]), "stack": parts[12], "tree": parts[13],
            })
    return rows

def kernels(db, t0, t1):
    con = sqlite3.connect(db)
    cur = con.cursor()
    out = []
    q = """SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN StringIds s ON k.demangledName=s.id
           WHERE k.start>=? AND k.end<=? ORDER BY k.start"""
    for start, end, name in cur.execute(q, (t0, t1)):
        out.append((start, end, name, "kernel"))
    q2 = """SELECT m.start,m.end,m.bytes FROM CUPTI_ACTIVITY_KIND_MEMSET m
            WHERE m.start>=? AND m.end<=? ORDER BY m.start"""
    for start, end, nbytes in cur.execute(q2, (t0, t1)):
        out.append((start, end, "[CUDA memset] %d B" % nbytes, "memset"))
    con.close()
    return out

def color(name):
    n = name.lower()
    if "spread_output_driven" in n: return "#e45756"
    if "calc_bin_size" in n: return "#f2a65a"
    if "calc_inverse" in n: return "#f6c177"
    if "calc_subprob" in n or "map_b_into" in n: return "#c9a0dc"
    if "uninitialized_fill" in n or "devicefor" in n: return "#9ecae1"
    if "scan" in n: return "#a1d99b"
    if name.startswith(":execute"): return "#59a14f"
    if name.startswith(":setpts"): return "#f28e2b"
    if name.startswith(":makeplan"): return "#4e79a7"
    if name.startswith(":iteration"): return "#bab0ac"
    if name.startswith("CCCL"): return "#b3cde3"
    return "#bdbdbd"

def short(name):
    name = name.replace("\n", " ")
    if "spread_output_driven" in name: return "spread_output_driven"
    if "calc_bin_size_noghost" in name: return "calc_bin_size_noghost"
    if "calc_inverse_of_global_sort_idx" in name: return "calc_inverse_of_global_sort_idx"
    if "calc_subprob" in name: return "calc_subprob"
    if "map_b_into_subprob" in name: return "map_b_into_subprob"
    if "uninitialized_fill_n" in name: return "CCCL uninitialized_fill_n"
    if "DeviceFor::Bulk" in name: return "CCCL DeviceFor::Bulk"
    if "inclusive_scan" in name: return "CCCL inclusive_scan"
    if "exclusive_scan" in name: return "CCCL exclusive_scan"
    if "DeviceScanInitKernel" in name: return "CCCL scan init"
    if "DeviceScanKernel" in name: return "CCCL scan"
    if "[CUDA memset]" in name: return name
    return name[:38]

def make_svg(nvtx_csv, db, out_svg, title, subtitle):
    rows = read_nvtx(nvtx_csv)
    iters = sorted([r for r in rows if r["name"].startswith(":iteration")], key=lambda r: r["dur"])
    it = iters[len(iters)//2]
    t0, t1 = it["start"], it["end"]
    dur = t1 - t0
    rr = [r for r in rows if r["start"] >= t0 and r["end"] <= t1]
    maxlvl = max(r["lvl"] for r in rr)
    W, left, right = 1400, 150, 30
    plotw = W - left - right
    top = 78
    row_h = 25
    cpu_h = (maxlvl + 1) * row_h
    gpu_top = top + cpu_h + 26
    gpu_h = 54
    H = gpu_top + gpu_h + 74

    def X(t):
        return left + (t - t0) / dur * plotw

    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="DejaVu Sans, Arial, sans-serif">')
    s.append(f'<rect width="{W}" height="{H}" fill="#fcfcfc"/>')
    s.append(f'<text x="{left}" y="28" font-size="20" font-weight="bold">{escape(title)}</text>')
    s.append(f'<text x="{left}" y="49" font-size="13" fill="#444">{escape(subtitle)}</text>')
    s.append(f'<text x="{left}" y="66" font-size="11" fill="#666">one representative iteration (median of {len(iters)}); x-axis: elapsed time within the iteration. CPU NVTX ranges (top), GPU activities (bottom).</text>')

    # grid lines
    for frac in [0, .25, .5, .75, 1.0]:
        t = t0 + frac * dur
        x = X(t)
        s.append(f'<line x1="{x:.1f}" y1="{top-4}" x2="{x:.1f}" y2="{gpu_top+gpu_h}" stroke="#e6e6e6" stroke-width="1"/>')
        s.append(f'<text x="{x:.1f}" y="{gpu_top+gpu_h+16}" font-size="10" fill="#555" text-anchor="middle">{frac*dur/1e6:.3f} ms</text>')

    # CPU flamegraph
    s.append(f'<text x="12" y="{top+16}" font-size="12" font-weight="bold" fill="#333">CPU / NVTX</text>')
    for r in rr:
        x0, x1 = X(r["start"]), X(r["end"])
        y = top + r["lvl"] * row_h
        w = max(1.0, x1 - x0)
        c = color(r["name"])
        s.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{w:.1f}" height="{row_h-3}" fill="{c}" stroke="#ffffff" stroke-width="0.7"><title>{escape(r["name"])} — {r["dur"]/1e6:.3f} ms</title></rect>')
        if w > 52:
            label = short(r["name"])
            s.append(f'<text x="{x0+3:.1f}" y="{y+15:.1f}" font-size="10" fill="#111">{escape(label)}</text>')

    # GPU lane
    s.append(f'<text x="12" y="{gpu_top+18}" font-size="12" font-weight="bold" fill="#333">GPU</text>')
    s.append(f'<rect x="{left}" y="{gpu_top+6}" width="{plotw}" height="{gpu_h-12}" fill="#f4f4f4" stroke="#ddd"/>')
    for start, end, name, kind in kernels(db, t0, t1):
        x0, x1 = X(start), X(end)
        w = max(1.5, x1 - x0)
        c = "#e45756" if "spread_output_driven" in name else ("#bdbdbd" if kind == "memset" else color(name))
        s.append(f'<rect x="{x0:.1f}" y="{gpu_top+11}" width="{w:.1f}" height="{gpu_h-22}" fill="{c}" stroke="#ffffff" stroke-width="0.5"><title>{escape(name)} — {(end-start)/1e6:.3f} ms</title></rect>')
        if w > 70:
            s.append(f'<text x="{x0+3:.1f}" y="{gpu_top+32}" font-size="10" fill="#111">{escape(short(name))}</text>')
    # annotate dominant kernel if too wide to read
    for start, end, name, kind in kernels(db, t0, t1):
        if "spread_output_driven" in name:
            s.append(f'<text x="{left+8}" y="{gpu_top+gpu_h+38}" font-size="12" fill="#b23a3a" font-weight="bold">spread_output_driven: {(end-start)/1e6:.3f} ms ({(end-start)/dur*100:.1f}% of iteration)</text>')
            break

    # legend
    leg = [(":iteration", "#bab0ac"), (":makeplan", "#4e79a7"), (":setpts", "#f28e2b"),
           (":execute", "#59a14f"), ("CCCL / CUB", "#9ecae1"), ("spread_output_driven", "#e45756")]
    lx = left
    ly = H - 24
    for name, c in leg:
        s.append(f'<rect x="{lx}" y="{ly-10}" width="12" height="12" fill="{c}"/>')
        s.append(f'<text x="{lx+16}" y="{ly}" font-size="10" fill="#444">{escape(name)}</text>')
        lx += 150
    s.append('</svg>')
    with open(out_svg, "w") as f:
        f.write("\n".join(s))
    print(out_svg, "iteration", it["start"], it["end"], "duration_ms", dur/1e6)
    for start, end, name, kind in kernels(db, t0, t1):
        if "spread_output_driven" in name:
            print("  spread kernel ms", (end-start)/1e6, "pct", (end-start)/dur*100)

if __name__ == "__main__":
    make_svg(*sys.argv[1:])
