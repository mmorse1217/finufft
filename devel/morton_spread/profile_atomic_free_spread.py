#!/usr/bin/env python3
"""Build, benchmark, profile, and summarize the atomic-free spread baseline.

The script is intentionally self-contained: it builds a line-info binary, runs
the benchmark, captures Nsight Compute reports for the four canonical workload
shapes, exports raw/detail/source views, and writes machine-readable CSVs plus a
human-readable Markdown digest.

Typical use:

    ./devel/morton_spread/profile_atomic_free_spread.py

Re-analyze an existing capture directory without touching the GPU:

    ./devel/morton_spread/profile_atomic_free_spread.py --analyze-only DIR

The output directory is printed at the end. Everything under it is disposable.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[2]

# Calibrated against the earlier manual captures. Each entry is the exact
# argument tail accepted by atomic_free_spread_bench, minus nrun.
CASES: Dict[str, Dict[str, object]] = {
    "uniform": {
        "args": [1_000_000, 64, 64, 64, 0, 4, 0, 42],
        "why": "dense uniform input; all output tiles have work",
    },
    "clustered": {
        "args": [1_000_000, 64, 64, 64, 1, 4, 0, 42],
        "why": "eight tight clusters; few hot output tiles",
    },
    "sparse": {
        "args": [100_000, 128, 128, 128, 0, 4, 0, 42],
        "why": "sparse uniform input on a larger grid",
    },
    "dense": {
        "args": [100_000, 64, 64, 64, 0, 4, 3, 42],
        "why": "one dense cluster; long same-tile accumulation",
    },
}

# Enough sections for the bottleneck analysis, without the expensive and
# largely irrelevant roofline/NVLink/NUMA/PM-sampling sections in --set full.
NCU_SECTIONS = [
    "SpeedOfLight",
    "LaunchStats",
    "Occupancy",
    "WorkloadDistribution",
    "ComputeWorkloadAnalysis",
    "MemoryWorkloadAnalysis",
    "MemoryWorkloadAnalysis_Tables",
    "SchedulerStats",
    "WarpStateStats",
    "InstructionStats",
    "SourceCounters",
]

# Heuristic grouping of source lines. The patterns are source-text based, so
# they survive line-number drift as long as the implementation keeps its shape.
SOURCE_GROUPS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "support index / modulo",
        re.compile(
            r"grid_size|coefficient_index|support_start|offset < WIDTH|offset < 0|"
            r"weight_index|covered|dest\[local\]"
        ),
    ),
    (
        "weight / select / accumulate",
        re.compile(r"float weight|real \+=|imag \+=|p\.strength|strength"),
    ),
    (
        "Horner / FMA",
        re.compile(
            r"eval_kernel_vec_horner|cudaFMA|__fmaf_rn|kx\[|ky\[|kz\[|"
            r"ker\[|c[0-5]\[i\]|horner"
        ),
    ),
    (
        "fold_rescale",
        re.compile(r"fold_rescale|floorf|__fsub_rd|__fmul_rd|static_cast<T>\(N\)"),
    ),
    ("ceilf", re.compile(r"ceilf")),
    (
        "bin iteration / offsets",
        re.compile(
            r"bin_offsets|for \(int .*bin|near_bins|x_bins|y_bins|z_bins|"
            r"floor_div|extended_bin_index|query_begin|query_end|local_bin|"
            r"tile_begin|tile_end|period"
        ),
    ),
    ("point load", re.compile(r"points\[i\]|Point p")),
    ("support branch", re.compile(r"if \(ix < 0|if \(iy < 0|continue;")),
]

OPCODE_RE = re.compile(r"^\s*(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)")
ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]+$")


@dataclass
class SourceRecord:
    """One deduplicated SASS PC and its first source correlation."""

    file: str
    line: Optional[int]
    source: str
    sass: str
    instructions: int
    thread_instructions: int
    predicated_thread_instructions: int
    samples: int
    not_issued_samples: int
    divergent_branches: int
    stalls: Dict[str, int] = field(default_factory=dict)


@dataclass
class SourceLine:
    file: str
    line: Optional[int]
    source: str
    instructions: int = 0
    thread_instructions: int = 0
    predicated_thread_instructions: int = 0
    samples: int = 0
    not_issued_samples: int = 0
    divergent_branches: int = 0
    pc_count: int = 0
    stalls: Dict[str, int] = field(default_factory=collections.Counter)

    @property
    def group(self) -> str:
        for name, pattern in SOURCE_GROUPS:
            if pattern.search(self.source):
                return name
        return "other / kernel setup"


# ---------------------------------------------------------------------------
# Process helpers


class CommandRunner:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lines: List[str] = []

    def note(self, text: str) -> None:
        print(text)
        self.lines.append(text)
        self.flush()

    def flush(self) -> None:
        self.log_path.write_text("\n".join(self.lines) + "\n")

    def command_text(self, cmd: Sequence[str]) -> str:
        return " ".join(shlex_quote(part) for part in cmd)

    def run(
        self,
        cmd: Sequence[str],
        *,
        log_name: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Path = REPO,
        check: bool = True,
        echo: bool = True,
    ) -> int:
        text = self.command_text(cmd)
        print(f"\n$ {text}")
        self.lines.append(f"\n$ {text}")
        self.flush()
        stdout_target = subprocess.PIPE
        log_file = None
        if log_name is not None:
            log_path = self.log_path.parent / log_name
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w")
            stdout_target = subprocess.PIPE
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=stdout_target,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if log_file is not None:
                log_file.write(line)
            if echo:
                print(line, end="")
        rc = proc.wait()
        if log_file is not None:
            log_file.close()
        if rc and check:
            raise RuntimeError(f"command failed with exit code {rc}: {text}")
        return rc

    def capture(
        self,
        cmd: Sequence[str],
        *,
        env: Optional[Dict[str, str]] = None,
        cwd: Path = REPO,
    ) -> str:
        text = self.command_text(cmd)
        self.lines.append(f"\n$ {text}")
        self.flush()
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.returncode:
            raise RuntimeError(
                f"command failed with exit code {proc.returncode}: {text}\n{proc.stdout}"
            )
        return proc.stdout


def shlex_quote(value: str) -> str:
    if not value:
        return "''"
    if re.fullmatch(r"[A-Za-z0-9_@%+=:,./-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def find_executable(name: str, candidates: Iterable[Path]) -> Optional[Path]:
    path = shutil.which(name)
    if path:
        return Path(path)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def detect_arch(nvidia_smi: Optional[Path]) -> str:
    if nvidia_smi is not None:
        try:
            out = subprocess.run(
                [
                    str(nvidia_smi),
                    "--query-gpu=compute_cap",
                    "--format=csv,noheader",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=True,
            ).stdout
            cap = first_line(out)
            match = re.fullmatch(r"(\d+)\.(\d+)", cap)
            if match:
                return f"sm_{match.group(1)}{match.group(2)}"
        except (OSError, subprocess.CalledProcessError):
            pass
    return "sm_75"


def default_cuda_home() -> Optional[Path]:
    candidates = []
    if os.environ.get("CUDA_HOME"):
        candidates.append(Path(os.environ["CUDA_HOME"]))
    if os.environ.get("CUDA_PATH"):
        candidates.append(Path(os.environ["CUDA_PATH"]))
    candidates.extend([Path.home() / "cuda-12.8", Path("/usr/local/cuda")])
    for candidate in candidates:
        if (candidate / "bin" / "nvcc").is_file():
            return candidate
    return None


def default_cccl() -> Optional[Path]:
    candidates = []
    if os.environ.get("CCCL"):
        candidates.append(Path(os.environ["CCCL"]))
    candidates.extend([Path("/tmp/cccl-3.4.2"), REPO / "build" / "gpu" / "_deps"])
    for candidate in candidates:
        if (candidate / "thrust").is_dir():
            return candidate
    return None


def build_environment(cuda_home: Path) -> Dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = str(cuda_home / "bin") + os.pathsep + env.get("PATH", "")
    lib64 = str(cuda_home / "lib64")
    env["LD_LIBRARY_PATH"] = lib64 + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


# ---------------------------------------------------------------------------
# Benchmark log parsing


def parse_bench_log(text: str) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for line in text.splitlines():
        if line.startswith("case "):
            result["case"] = line[len("case ") :].strip()
        elif line.startswith("prep "):
            result["prep"] = line[len("prep ") :].strip()
        elif line.startswith("ref "):
            result["reference"] = line[len("ref ") :].strip()
            match = re.search(
                r"setpts=([\d.]+) ms execute=([\d.]+) ms total=([\d.]+) ms", line
            )
            if match:
                result["reference_setpts_ms"] = float(match.group(1))
                result["reference_execute_ms"] = float(match.group(2))
                result["reference_total_ms"] = float(match.group(3))
        elif line.startswith("spread warmup="):
            result["spread"] = line[len("spread ") :].strip()
            match = re.search(
                r"min=([\d.]+) median=([\d.]+) max=([\d.]+) ms "
                r"throughput=([\d.]+) Mpoint/s ([\d.]+) Mcell/s",
                line,
            )
            if match:
                result["spread_min_ms"] = float(match.group(1))
                result["spread_median_ms"] = float(match.group(2))
                result["spread_max_ms"] = float(match.group(3))
                result["spread_Mpoint_s"] = float(match.group(4))
                result["spread_Mcell_s"] = float(match.group(5))
        elif line.startswith("timing "):
            result["timing"] = line[len("timing ") :].strip()
            match = re.search(
                r"setup=([\d.]+) ms end_to_end=([\d.]+) ms "
                r"spread_vs_ref_exec=([\d.]+)x",
                line,
            )
            if match:
                result["setup_ms"] = float(match.group(1))
                result["end_to_end_ms"] = float(match.group(2))
                result["spread_vs_ref_exec"] = float(match.group(3))
        elif line.startswith("correct "):
            result["correct"] = line[len("correct ") :].strip()
            match = re.search(
                r"relL2=([\d.eE+-]+) max_abs=([\d.eE+-]+) "
                r"max_rel=([\d.eE+-]+) total_rel=([\d.eE+-]+) "
                r"unwritten=(\d+) nonfinite=(\d+)",
                line,
            )
            if match:
                result["relL2"] = float(match.group(1))
                result["max_abs"] = float(match.group(2))
                result["max_rel"] = float(match.group(3))
                result["total_rel"] = float(match.group(4))
                result["unwritten"] = int(match.group(5))
                result["nonfinite"] = int(match.group(6))
        elif line.startswith("check "):
            result["check"] = line[len("check ") :].strip()
            match = re.search(r"overall=(\w+)", line)
            if match:
                result["overall"] = match.group(1)
    return result


# ---------------------------------------------------------------------------
# NCU export parsing


def read_csv_rows(path: Path) -> List[List[str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.reader(handle))


def parse_details(path: Path) -> List[Dict[str, str]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8", errors="replace")))
    if not rows:
        return []
    kernel_names = {row.get("Kernel Name", "") for row in rows}
    chosen = next((name for name in kernel_names if "spread" in name), None)
    if chosen is None:
        chosen = sorted(kernel_names)[0]
    return [row for row in rows if row.get("Kernel Name", "") == chosen]


def metric_map(details: List[Dict[str, str]]) -> Dict[Tuple[str, str, str], str]:
    result: Dict[Tuple[str, str, str], str] = {}
    for row in details:
        key = (
            row.get("Section Name", ""),
            row.get("Body Item Label", ""),
            row.get("Metric Name", ""),
        )
        result.setdefault(key, row.get("Metric Value", ""))
    return result


def metric_value(
    metrics: Dict[Tuple[str, str, str], str],
    section: str,
    name: str,
    body: str = "",
) -> Optional[str]:
    value = metrics.get((section, body, name))
    if value is None:
        # Be tolerant of section spelling / capitalization changes between ncu
        # releases: fall back to a unique metric-name match.
        matches = [
            v
            for (sec, bod, metric), v in metrics.items()
            if metric == name and (not section or sec == section) and (not body or bod == body)
        ]
        if len(set(matches)) == 1:
            return matches[0]
        return None
    return value


def numeric(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip().replace(",", "")
    if not value or value == "-":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_raw_csv(path: Path) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
    rows = read_csv_rows(path)
    if not rows:
        return [], [], []
    header_index = next(
        (i for i, row in enumerate(rows) if row and row[0].strip() == "ID"), 0
    )
    names = rows[header_index]
    units = rows[header_index + 1] if header_index + 1 < len(rows) else [""] * len(names)
    records: List[Dict[str, str]] = []
    for row in rows[header_index + 2 :]:
        if not row or not row[0].strip():
            continue
        for index, name in enumerate(names):
            if not name or name == "ID":
                continue
            value = row[index] if index < len(row) else ""
            records.append(
                {
                    "kernel_id": row[0] if row else "",
                    "metric": name,
                    "unit": units[index] if index < len(units) else "",
                    "value": value,
                }
            )
    return names, units, records


def parse_number(row: Sequence[str], columns: Dict[str, int], name: str) -> int:
    index = columns.get(name)
    if index is None or index >= len(row):
        return 0
    value = row[index].strip()
    if value in ("", "-"):
        return 0
    value = value.replace(",", "")
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            return 0


def parse_source_csv(path: Path) -> Tuple[List[SourceRecord], Dict[str, int]]:
    """Deduplicate SASS PCs and keep the first source correlation.

    Nsight repeats the same PC under inline definitions and call sites. Those
    rows are correlations, not independent executions, so counting them would
    inflate both instructions and samples.
    """
    current_file = ""
    current_line: Optional[int] = None
    current_source = ""
    current_kernel = ""
    columns: Optional[Dict[str, int]] = None
    row_width = 0
    seen: Dict[Tuple[str, int], SourceRecord] = {}
    duplicates = 0
    conflicting = 0

    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if row[0] == "File Path":
                current_file = row[1] if len(row) > 1 else ""
                current_line = None
                current_source = ""
                continue
            if row[0] == "Function Name":
                current_kernel = row[1] if len(row) > 1 else ""
                current_line = None
                current_source = ""
                continue
            if row[0] == "Line No":
                columns = {name: i for i, name in enumerate(row)}
                row_width = len(row)
                continue
            if columns is None or len(row) != row_width:
                continue
            # The reference library also exports kernels whose names contain
            # "spread" (for example spread_output_driven). Keep only this
            # translation unit's kernel.
            if "atomic_free_spread" not in current_kernel or "assign_bins" in current_kernel:
                continue
            if row[0].strip():
                text = row[0].strip()
                current_line = int(text) if text.isdigit() else None
                current_source = row[1].strip() if len(row) > 1 else ""
                # Line-aggregate rows use "-" as the address; PC rows follow.
            address_text = row[columns["Address"]].strip() if "Address" in columns else ""
            if not ADDRESS_RE.fullmatch(address_text):
                continue
            address = int(address_text, 16)
            key = (current_kernel, address)
            sass = row[columns["Source"]].strip() if "Source" in columns else ""
            stalls: Dict[str, int] = {}
            for name, index in columns.items():
                if name.startswith("stall_") and "(Not Issued)" not in name:
                    stalls[name[len("stall_") :]] = parse_number(row, columns, name)
            record = SourceRecord(
                file=current_file,
                line=current_line,
                source=current_source,
                sass=sass,
                instructions=parse_number(row, columns, "Instructions Executed"),
                thread_instructions=parse_number(
                    row, columns, "Thread Instructions Executed"
                ),
                predicated_thread_instructions=parse_number(
                    row, columns, "Predicated-On Thread Instructions Executed"
                ),
                samples=parse_number(
                    row, columns, "Warp Stall Sampling (All Samples)"
                ),
                not_issued_samples=parse_number(
                    row, columns, "Warp Stall Sampling (Not-issued Samples)"
                ),
                divergent_branches=parse_number(row, columns, "Divergent Branches"),
                stalls=stalls,
            )
            previous = seen.get(key)
            if previous is not None:
                duplicates += 1
                if previous.instructions != record.instructions:
                    conflicting += 1
                continue
            seen[key] = record

    return list(seen.values()), {
        "unique_pcs": len(seen),
        "duplicate_pc_rows": duplicates,
        "conflicting_pc_rows": conflicting,
    }


def aggregate_source_lines(records: Sequence[SourceRecord]) -> List[SourceLine]:
    lines: Dict[Tuple[str, Optional[int], str], SourceLine] = {}
    for record in records:
        file_name = Path(record.file).name if record.file else "unmapped"
        key = (file_name, record.line, record.source)
        line = lines.setdefault(
            key,
            SourceLine(
                file=file_name,
                line=record.line,
                source=record.source or "(no source correlation)",
            ),
        )
        line.instructions += record.instructions
        line.thread_instructions += record.thread_instructions
        line.predicated_thread_instructions += record.predicated_thread_instructions
        line.samples += record.samples
        line.not_issued_samples += record.not_issued_samples
        line.divergent_branches += record.divergent_branches
        line.pc_count += 1
        line.stalls.update(record.stalls)
    return sorted(lines.values(), key=lambda item: item.instructions, reverse=True)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def percent(value: float, total: float) -> float:
    return 0.0 if not total else 100.0 * value / total


# ---------------------------------------------------------------------------
# Analysis


def analyze_case(
    case: str,
    out_dir: Path,
    bench: Dict[str, object],
    details_path: Path,
    source_path: Path,
) -> Dict[str, object]:
    analysis_dir = out_dir / "analysis" / case
    analysis_dir.mkdir(parents=True, exist_ok=True)

    details = parse_details(details_path)
    metrics = metric_map(details)
    source_records, source_stats = parse_source_csv(source_path)
    source_lines = aggregate_source_lines(source_records)

    total_inst = sum(line.instructions for line in source_lines)
    total_samples = sum(line.samples for line in source_lines)
    source_stats["source_instructions"] = total_inst
    source_stats["source_samples"] = total_samples

    # Raw detail table: every metric ncu exported, grouped by section.
    write_csv(
        analysis_dir / "metrics_details.csv",
        ["section", "body_item", "metric", "unit", "value"],
        [
            {
                "section": row.get("Section Name", ""),
                "body_item": row.get("Body Item Label", ""),
                "metric": row.get("Metric Name", ""),
                "unit": row.get("Metric Unit", ""),
                "value": row.get("Metric Value", ""),
            }
            for row in details
        ],
    )

    # Stalls: Warp State Statistics gives cycles-per-issued-instruction by
    # reason. Convert both to a share of all warp-state cycles and of stalls.
    stall_rows = [
        row
        for row in details
        if row.get("Section Name") == "Warp State Statistics"
        and row.get("Body Item Label") == "Warp State (All Cycles)"
    ]
    stall_values: Dict[str, float] = {}
    for row in stall_rows:
        value = numeric(row.get("Metric Value"))
        if value is not None:
            stall_values[row.get("Metric Name", "")] = value
    warp_state_total = sum(stall_values.values())
    stall_total = sum(
        value for name, value in stall_values.items() if name.startswith("Stall ")
    )

    def stall_share(name: str) -> Optional[float]:
        value = stall_values.get(name)
        if value is None or not warp_state_total:
            return None
        return percent(value, warp_state_total)

    write_csv(
        analysis_dir / "stalls.csv",
        [
            "reason",
            "cycles_per_issued_instruction",
            "share_of_warp_state_pct",
            "share_of_stall_cycles_pct",
        ],
        [
            {
                "reason": name,
                "cycles_per_issued_instruction": f"{value:.6f}",
                "share_of_warp_state_pct": f"{percent(value, warp_state_total):.3f}",
                "share_of_stall_cycles_pct": (
                    f"{percent(value, stall_total):.3f}"
                    if name.startswith("Stall ")
                    else "-"
                ),
            }
            for name, value in sorted(
                stall_values.items(), key=lambda item: item[1], reverse=True
            )
        ],
    )

    # Source hotspots.
    hotspot_rows = []
    for line in source_lines:
        top_stall_name, top_stall_value = (
            max(line.stalls.items(), key=lambda item: item[1])
            if line.stalls
            else ("", 0)
        )
        hotspot_rows.append(
            {
                "file": line.file,
                "line": "" if line.line is None else line.line,
                "source": line.source,
                "instructions": line.instructions,
                "instruction_pct": f"{percent(line.instructions, total_inst):.4f}",
                "samples": line.samples,
                "sample_pct": f"{percent(line.samples, total_samples):.4f}",
                "thread_instructions": line.thread_instructions,
                "predicated_thread_instructions": line.predicated_thread_instructions,
                "not_issued_samples": line.not_issued_samples,
                "divergent_branches": line.divergent_branches,
                "pc_count": line.pc_count,
                "source_group": line.group,
                "top_stall": top_stall_name,
                "top_stall_samples": top_stall_value,
            }
        )
    write_csv(
        analysis_dir / "source_hotspots.csv",
        [
            "file",
            "line",
            "source",
            "instructions",
            "instruction_pct",
            "samples",
            "sample_pct",
            "thread_instructions",
            "predicated_thread_instructions",
            "not_issued_samples",
            "divergent_branches",
            "pc_count",
            "source_group",
            "top_stall",
            "top_stall_samples",
        ],
        hotspot_rows,
    )

    # Source line x stall reason, so source attribution can be inspected
    # without re-exporting the report.
    source_stall_rows = []
    for line in source_lines:
        for reason, value in sorted(
            line.stalls.items(), key=lambda item: item[1], reverse=True
        ):
            if value <= 0:
                continue
            source_stall_rows.append(
                {
                    "file": line.file,
                    "line": "" if line.line is None else line.line,
                    "source": line.source,
                    "source_group": line.group,
                    "stall_reason": reason,
                    "samples": value,
                    "share_of_line_samples_pct": f"{percent(value, line.samples):.3f}",
                }
            )
    write_csv(
        analysis_dir / "source_stalls.csv",
        [
            "file",
            "line",
            "source",
            "source_group",
            "stall_reason",
            "samples",
            "share_of_line_samples_pct",
        ],
        source_stall_rows,
    )

    # Semantic source groups.
    group_totals: Dict[str, Dict[str, float]] = collections.defaultdict(
        lambda: {"instructions": 0, "samples": 0, "lines": 0, "pcs": 0}
    )
    for line in source_lines:
        group = group_totals[line.group]
        group["instructions"] += line.instructions
        group["samples"] += line.samples
        group["lines"] += 1
        group["pcs"] += line.pc_count
    source_group_rows = [
        {
            "source_group": name,
            "instructions": int(values["instructions"]),
            "instruction_pct": f"{percent(values['instructions'], total_inst):.3f}",
            "samples": int(values["samples"]),
            "sample_pct": f"{percent(values['samples'], total_samples):.3f}",
            "source_lines": int(values["lines"]),
            "sass_pcs": int(values["pcs"]),
        }
        for name, values in sorted(
            group_totals.items(), key=lambda item: item[1]["instructions"], reverse=True
        )
    ]
    write_csv(
        analysis_dir / "source_groups.csv",
        [
            "source_group",
            "instructions",
            "instruction_pct",
            "samples",
            "sample_pct",
            "source_lines",
            "sass_pcs",
        ],
        source_group_rows,
    )

    # SASS opcode mix, exact and family-level.
    opcode_totals: Dict[str, Dict[str, float]] = collections.defaultdict(
        lambda: {"instructions": 0, "samples": 0, "pcs": 0}
    )
    family_totals: Dict[str, Dict[str, float]] = collections.defaultdict(
        lambda: {"instructions": 0, "samples": 0, "pcs": 0}
    )
    for record in source_records:
        match = OPCODE_RE.match(record.sass)
        if not match:
            continue
        opcode = match.group(1)
        family = opcode.split(".", 1)[0]
        for table, name in ((opcode_totals, opcode), (family_totals, family)):
            entry = table[name]
            entry["instructions"] += record.instructions
            entry["samples"] += record.samples
            entry["pcs"] += 1
    opcode_rows = [
        {
            "opcode": name,
            "instructions": int(values["instructions"]),
            "instruction_pct": f"{percent(values['instructions'], total_inst):.3f}",
            "samples": int(values["samples"]),
            "sample_pct": f"{percent(values['samples'], total_samples):.3f}",
            "sass_pcs": int(values["pcs"]),
        }
        for name, values in sorted(
            opcode_totals.items(), key=lambda item: item[1]["instructions"], reverse=True
        )
    ]
    family_rows = [
        {
            "opcode_family": name,
            "instructions": int(values["instructions"]),
            "instruction_pct": f"{percent(values['instructions'], total_inst):.3f}",
            "samples": int(values["samples"]),
            "sample_pct": f"{percent(values['samples'], total_samples):.3f}",
            "sass_pcs": int(values["pcs"]),
        }
        for name, values in sorted(
            family_totals.items(), key=lambda item: item[1]["instructions"], reverse=True
        )
    ]
    write_csv(
        analysis_dir / "sass_opcodes.csv",
        [
            "opcode",
            "instructions",
            "instruction_pct",
            "samples",
            "sample_pct",
            "sass_pcs",
        ],
        opcode_rows,
    )
    write_csv(
        analysis_dir / "sass_opcode_families.csv",
        [
            "opcode_family",
            "instructions",
            "instruction_pct",
            "samples",
            "sample_pct",
            "sass_pcs",
        ],
        family_rows,
    )

    # Key metrics used in the cross-case table.
    key: Dict[str, object] = {}
    definitions = [
        ("duration_ms", "GPU Speed Of Light Throughput", "Duration", ""),
        ("compute_pct", "GPU Speed Of Light Throughput", "Compute (SM) Throughput", ""),
        ("memory_pct", "GPU Speed Of Light Throughput", "Memory Throughput", ""),
        ("dram_pct", "GPU Speed Of Light Throughput", "DRAM Throughput", ""),
        ("l1tex_pct", "GPU Speed Of Light Throughput", "L1/TEX Cache Throughput", ""),
        ("l2_pct", "GPU Speed Of Light Throughput", "L2 Cache Throughput", ""),
        ("elapsed_cycles", "GPU Speed Of Light Throughput", "Elapsed Cycles", ""),
        ("waves_per_sm", "GPU Speed Of Light Throughput", "Waves Per SM", ""),
        ("ipc", "Compute Workload Analysis", "Executed Ipc Active", ""),
        ("issue_slots_busy_pct", "Compute Workload Analysis", "Issue Slots Busy", ""),
        ("sm_busy_pct", "Compute Workload Analysis", "SM Busy", ""),
        (
            "fma_pipe_pct",
            "Compute Workload Analysis",
            "FMA",
            "Pipe Utilization (% of active cycles)",
        ),
        (
            "alu_pipe_pct",
            "Compute Workload Analysis",
            "ALU",
            "Pipe Utilization (% of active cycles)",
        ),
        (
            "xu_pipe_pct",
            "Compute Workload Analysis",
            "XU",
            "Pipe Utilization (% of peak instructions executed)",
        ),
        (
            "l1_hit_global_loads_pct",
            "Memory Workload Analysis Tables",
            "L1TEX Hit Rate For Global Loads",
            "",
        ),
        (
            "l2_hit_loads_pct",
            "Memory Workload Analysis Tables",
            "L2 Hit Rate For Loads",
            "",
        ),
        (
            "avg_bytes_per_sector_load",
            "Memory Workload Analysis Tables",
            "Average Bytes Per Sector For Global Loads",
            "",
        ),
        (
            "no_eligible_pct",
            "Scheduler Statistics",
            "No Eligible",
            "",
        ),
        (
            "active_warps_per_scheduler",
            "Scheduler Statistics",
            "Active Warps Per Scheduler",
            "",
        ),
        (
            "eligible_warps_per_scheduler",
            "Scheduler Statistics",
            "Eligible Warps Per Scheduler",
            "",
        ),
        (
            "warp_cycles_per_issued_inst",
            "Warp State Statistics",
            "Warp Cycles Per Issued Instruction",
            "",
        ),
        (
            "active_threads_per_warp",
            "Warp State Statistics",
            "Avg. Active Threads Per Warp",
            "",
        ),
        (
            "predicated_threads_per_warp",
            "Warp State Statistics",
            "Avg. Not Predicated Off Threads Per Warp",
            "",
        ),
        (
            "instructions_executed",
            "Instruction Statistics",
            "Instructions Executed",
            "",
        ),
        (
            "branch_instructions_ratio_pct",
            "Source Counters",
            "Branch Instructions Ratio",
            "",
        ),
        (
            "theoretical_occupancy_pct",
            "Occupancy",
            "Theoretical Occupancy",
            "",
        ),
        ("achieved_occupancy_pct", "Occupancy", "Achieved Occupancy", ""),
        (
            "registers_per_thread",
            "Launch Statistics",
            "Registers Per Thread",
            "",
        ),
        (
            "static_shared_bytes",
            "Launch Statistics",
            "Static Shared Memory Per Block",
            "",
        ),
        ("grid_size", "Launch Statistics", "Grid Size", ""),
        ("block_size", "Launch Statistics", "Block Size", ""),
        ("threads", "Launch Statistics", "Threads", ""),
    ]
    for key_name, section, metric, body in definitions:
        raw = metric_value(metrics, section, metric, body)
        key[key_name] = numeric(raw) if numeric(raw) is not None else raw
    for reason in [
        "Stall Not Selected",
        "Stall Math Pipe Throttle",
        "Stall Wait",
        "Stall Long Scoreboard",
        "Stall MIO Throttle",
        "Stall Short Scoreboard",
        "Stall Barrier",
        "Stall Dispatch Stall",
        "Stall Branch Resolving",
        "Stall No Instruction",
        "Stall Drain",
        "Stall Imc Miss",
        "Stall Lg Throttle",
        "Stall Tex Throttle",
        "Stall Sleeping",
    ]:
        key[reason.lower().replace(" ", "_") + "_share_pct"] = stall_share(reason)

    key["source_instruction_total"] = total_inst
    key["source_sample_total"] = total_samples
    key["source_unique_pcs"] = source_stats["unique_pcs"]
    key["source_duplicate_pc_rows"] = source_stats["duplicate_pc_rows"]
    key["source_conflicting_pc_rows"] = source_stats["conflicting_pc_rows"]
    if source_lines:
        top = source_lines[0]
        key["top_source_file"] = top.file
        key["top_source_line"] = top.line
        key["top_source_text"] = top.source
        key["top_source_instruction_pct"] = percent(top.instructions, total_inst)
    if hotspot_rows:
        top_sample_row = max(hotspot_rows, key=lambda item: item["samples"])
        key["top_sample_file"] = top_sample_row["file"]
        key["top_sample_line"] = top_sample_row["line"]
        key["top_sample_text"] = top_sample_row["source"]
        key["top_sample_pct"] = top_sample_row["sample_pct"]
    if stall_values:
        top_stall = max(stall_values.items(), key=lambda item: item[1])
        key["top_stall"] = top_stall[0]
        key["top_stall_cycles_per_issued_inst"] = top_stall[1]
        key["top_stall_share_pct"] = stall_share(top_stall[0])

    # Keep the Markdown digest small but informative.
    top_source_lines = [
        {
            "file": row["file"],
            "line": row["line"],
            "source": row["source"],
            "instructions": row["instructions"],
            "instruction_pct": row["instruction_pct"],
            "samples": row["samples"],
            "sample_pct": row["sample_pct"],
            "pc_count": row["pc_count"],
            "group": row["source_group"],
            "top_stall": row["top_stall"],
        }
        for row in hotspot_rows[:20]
    ]
    top_sample_lines = [
        {
            "file": row["file"],
            "line": row["line"],
            "source": row["source"],
            "instructions": row["instructions"],
            "instruction_pct": row["instruction_pct"],
            "samples": row["samples"],
            "sample_pct": row["sample_pct"],
            "group": row["source_group"],
            "top_stall": row["top_stall"],
        }
        for row in sorted(
            hotspot_rows, key=lambda item: item["samples"], reverse=True
        )[:15]
    ]

    return {
        "case": case,
        "args": CASES.get(case, {}).get("args"),
        "why": CASES.get(case, {}).get("why", ""),
        "bench": bench,
        "key_metrics": key,
        "stalls": [
            {
                "reason": row["reason"],
                "cycles_per_issued_instruction": row[
                    "cycles_per_issued_instruction"
                ],
                "share_of_warp_state_pct": row["share_of_warp_state_pct"],
                "share_of_stall_cycles_pct": row["share_of_stall_cycles_pct"],
            }
            for row in sorted(
                [
                    {
                        "reason": name,
                        "cycles_per_issued_instruction": f"{value:.6f}",
                        "share_of_warp_state_pct": f"{percent(value, warp_state_total):.3f}",
                        "share_of_stall_cycles_pct": (
                            f"{percent(value, stall_total):.3f}"
                            if name.startswith("Stall ")
                            else "-"
                        ),
                    }
                    for name, value in stall_values.items()
                ],
                key=lambda item: float(item["share_of_warp_state_pct"]),
                reverse=True,
            )
        ],
        "source_groups": source_group_rows,
        "top_source_lines": top_source_lines,
        "top_sample_lines": top_sample_lines,
        "top_opcodes": opcode_rows[:30],
        "top_opcode_families": family_rows[:20],
        "source_stats": source_stats,
        "artifacts": {
            "report": f"reports/{case}.ncu-rep",
            "raw_csv": f"ncu/{case}.raw.csv",
            "details_csv": f"ncu/{case}.details.csv",
            "details_txt": f"ncu/{case}.details.txt",
            "source_csv": f"ncu/{case}.source.csv",
            "source_txt": f"ncu/{case}.source.txt",
            "metrics_details_csv": f"analysis/{case}/metrics_details.csv",
            "stalls_csv": f"analysis/{case}/stalls.csv",
            "source_hotspots_csv": f"analysis/{case}/source_hotspots.csv",
            "source_stalls_csv": f"analysis/{case}/source_stalls.csv",
            "source_groups_csv": f"analysis/{case}/source_groups.csv",
            "sass_opcodes_csv": f"analysis/{case}/sass_opcodes.csv",
            "sass_opcode_families_csv": f"analysis/{case}/sass_opcode_families.csv",
        },
    }


# ---------------------------------------------------------------------------
# Markdown / JSON output


def fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if digits == 0 and value.is_integer():
            return f"{int(value):,}"
        if abs(value) >= 1e6 or (abs(value) < 1e-3 and value != 0):
            return f"{value:.{digits}e}"
        return f"{value:.{digits}f}"
    return str(value)


def md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def truncate(text: str, width: int = 56) -> str:
    text = text.replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def write_summary(
    out_dir: Path,
    environment: Dict[str, str],
    results: Dict[str, Dict[str, object]],
    order: Sequence[str],
) -> None:
    lines: List[str] = []
    lines.append("# Atomic-free spread profiling digest")
    lines.append("")
    lines.append(f"Generated: {environment.get('generated_at', '-')}")
    lines.append("")
    lines.append(
        f"Repository: `{environment.get('repo', '-')}`  \n"
        f"Commit: `{environment.get('git_commit', '-')}`  \n"
        f"Git status: {environment.get('git_status', '-')}  \n"
        f"GPU: {environment.get('gpu', '-')}  \n"
        f"CUDA: {environment.get('nvcc', '-')}  \n"
        f"Nsight Compute: {environment.get('ncu', '-')}  \n"
        f"Build: `{environment.get('build_command', '-')}`"
    )
    lines.append("")
    lines.append("## Artifact map")
    lines.append("")
    lines.append(
        "- `reports/<case>.ncu-rep` — original Nsight Compute report\n"
        "- `ncu/<case>.raw.csv` — every raw metric/unit/value from the report\n"
        "- `ncu/<case>.details.csv` / `.txt` — section-organized metrics and rules\n"
        "- `ncu/<case>.source.csv` / `.txt` — CUDA/SASS source correlation\n"
        "- `analysis/<case>/source_hotspots.csv` — every source line by instructions/samples\n"
        "- `analysis/<case>/source_stalls.csv` — source line x stall reason\n"
        "- `analysis/<case>/source_groups.csv` — heuristic semantic grouping\n"
        "- `analysis/<case>/sass_opcodes.csv` / `sass_opcode_families.csv` — SASS mix\n"
        "- `analysis/<case>/stalls.csv` — warp-state stall breakdown\n"
        "- `analysis/<case>/metrics_details.csv` — all detail metrics in long form\n"
    )
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    case_rows = []
    for case in order:
        if case not in results:
            continue
        result = results[case]
        bench = result["bench"]
        key = result["key_metrics"]
        case_rows.append(
            [
                case,
                " ".join(str(v) for v in result["args"]),
                fmt(bench.get("spread_median_ms")),
                fmt(key.get("duration_ms")),
                fmt(bench.get("relL2"), 3),
                bench.get("overall", "-"),
            ]
        )
    lines.append(
        md_table(
            ["case", "bench args (M N1 N2 N3 cluster bin_size pattern seed)", "bench median ms", "ncu duration ms", "relL2", "checks"],
            case_rows,
        )
    )
    lines.append("")
    lines.append(
        "NCU durations come from one replayed launch with unlocked clocks; they are not "
        "benchmark medians. Use utilization/instruction/stall columns for comparison, "
        "and the benchmark columns for wall-clock."
    )
    lines.append("")

    lines.append("## Cross-case key metrics")
    lines.append("")
    cross_headers = [
        "metric",
        *[case for case in order if case in results],
    ]
    cross_metrics = [
        ("ncu duration (ms)", "duration_ms", 2),
        ("compute (SM) %", "compute_pct", 2),
        ("memory %", "memory_pct", 2),
        ("DRAM %", "dram_pct", 2),
        ("L1/TEX %", "l1tex_pct", 2),
        ("L2 %", "l2_pct", 2),
        ("IPC", "ipc", 2),
        ("issue slots busy %", "issue_slots_busy_pct", 2),
        ("FMA pipe %", "fma_pipe_pct", 2),
        ("ALU pipe %", "alu_pipe_pct", 2),
        ("XU pipe %", "xu_pipe_pct", 2),
        ("achieved occupancy %", "achieved_occupancy_pct", 2),
        ("active warps/scheduler", "active_warps_per_scheduler", 2),
        ("eligible warps/scheduler", "eligible_warps_per_scheduler", 2),
        ("no eligible %", "no_eligible_pct", 2),
        ("warp cycles / issued inst", "warp_cycles_per_issued_inst", 2),
        ("active threads / warp", "active_threads_per_warp", 2),
        ("predicated threads / warp", "predicated_threads_per_warp", 2),
        ("branch inst ratio %", "branch_instructions_ratio_pct", 2),
        ("L1 hit global loads %", "l1_hit_global_loads_pct", 2),
        ("L2 hit loads %", "l2_hit_loads_pct", 2),
        ("avg bytes/sector load", "avg_bytes_per_sector_load", 2),
        ("warp instructions", "instructions_executed", 0),
        ("registers/thread", "registers_per_thread", 0),
        ("static shared bytes", "static_shared_bytes", 0),
        ("grid blocks", "grid_size", 0),
        ("block threads", "block_size", 0),
        ("source instruction total", "source_instruction_total", 0),
        ("source sample total", "source_sample_total", 0),
    ]
    cross_rows = []
    for label, key_name, digits in cross_metrics:
        cross_rows.append(
            [
                label,
                *[fmt(results[case]["key_metrics"].get(key_name), digits) for case in order if case in results],
            ]
        )
    lines.append(md_table(cross_headers, cross_rows))
    lines.append("")

    for case in order:
        if case not in results:
            continue
        result = results[case]
        bench = result["bench"]
        key = result["key_metrics"]
        lines.append(f"## Case: `{case}`")
        lines.append("")
        lines.append(f"{result.get('why', '')}")
        lines.append("")
        lines.append(
            f"Benchmark: median **{fmt(bench.get('spread_median_ms'))} ms** "
            f"(min {fmt(bench.get('spread_min_ms'))}, max {fmt(bench.get('spread_max_ms'))}), "
            f"relL2 **{fmt(bench.get('relL2'), 3)}**, checks **{bench.get('overall', '-')}**. "
            f"Method 3 execute: {fmt(bench.get('reference_execute_ms'))} ms. "
            f"Spread / method-3 execute: {fmt(bench.get('spread_vs_ref_exec'))}x."
        )
        lines.append("")
        lines.append(
            f"NCU: **{fmt(key.get('duration_ms'))} ms**, compute **{fmt(key.get('compute_pct'))}%**, "
            f"memory **{fmt(key.get('memory_pct'))}%**, DRAM **{fmt(key.get('dram_pct'))}%**, "
            f"IPC **{fmt(key.get('ipc'))}**, occupancy **{fmt(key.get('achieved_occupancy_pct'))}%**, "
            f"no-eligible **{fmt(key.get('no_eligible_pct'))}%**."
        )
        lines.append("")

        # Stalls.
        lines.append("### Stall breakdown")
        lines.append("")
        stall_rows = [
            [
                row["reason"],
                fmt(row["cycles_per_issued_instruction"], 3),
                fmt(row["share_of_warp_state_pct"], 2) + "%",
                fmt(row["share_of_stall_cycles_pct"], 2) + "%",
            ]
            for row in result["stalls"][:12]
        ]
        lines.append(
            md_table(
                [
                    "warp state",
                    "cycles / issued inst",
                    "share of warp state",
                    "share of stalls",
                ],
                stall_rows,
            )
        )
        lines.append("")

        # Source groups.
        lines.append("### Source groups (heuristic)")
        lines.append("")
        group_rows = [
            [
                row["source_group"],
                fmt(row["instructions"], 0),
                fmt(row["instruction_pct"], 2) + "%",
                fmt(row["samples"], 0),
                fmt(row["sample_pct"], 2) + "%",
                row["source_lines"],
                row["sass_pcs"],
            ]
            for row in result["source_groups"]
        ]
        lines.append(
            md_table(
                [
                    "group",
                    "instructions",
                    "inst %",
                    "samples",
                    "sample %",
                    "source lines",
                    "SASS PCs",
                ],
                group_rows,
            )
        )
        lines.append("")

        # Top source lines by instructions.
        lines.append("### Top source lines by instructions")
        lines.append("")
        top_rows = [
            [
                f"`{row['file']}:{row['line']}`",
                truncate(row["source"], 52),
                fmt(row["instructions"], 0),
                fmt(row["instruction_pct"], 2) + "%",
                fmt(row["samples"], 0),
                fmt(row["sample_pct"], 2) + "%",
                row["group"],
                row["top_stall"],
            ]
            for row in result["top_source_lines"][:15]
        ]
        lines.append(
            md_table(
                [
                    "site",
                    "source",
                    "instructions",
                    "inst %",
                    "samples",
                    "sample %",
                    "group",
                    "top stall",
                ],
                top_rows,
            )
        )
        lines.append("")

        # Top source lines by samples.
        lines.append("### Top source lines by PC samples")
        lines.append("")
        sample_rows = [
            [
                f"`{row['file']}:{row['line']}`",
                truncate(row["source"], 52),
                fmt(row["samples"], 0),
                fmt(row["sample_pct"], 2) + "%",
                fmt(row["instructions"], 0),
                fmt(row["instruction_pct"], 2) + "%",
                row["top_stall"],
            ]
            for row in result["top_sample_lines"][:12]
        ]
        lines.append(
            md_table(
                ["site", "source", "samples", "sample %", "instructions", "inst %", "top stall"],
                sample_rows,
            )
        )
        lines.append("")

        # Opcode families.
        lines.append("### Top SASS opcode families")
        lines.append("")
        opcode_rows = [
            [
                row["opcode_family"],
                fmt(row["instructions"], 0),
                fmt(row["instruction_pct"], 2) + "%",
                fmt(row["samples"], 0),
                fmt(row["sample_pct"], 2) + "%",
                row["sass_pcs"],
            ]
            for row in result["top_opcode_families"][:15]
        ]
        lines.append(
            md_table(
                ["family", "instructions", "inst %", "samples", "sample %", "SASS PCs"],
                opcode_rows,
            )
        )
        lines.append("")
        lines.append("### Artifacts for this case")
        lines.append("")
        for label, relative in result["artifacts"].items():
            lines.append(f"- `{relative}` ({label})")
        lines.append("")

    lines.append("## Method notes")
    lines.append("")
    lines.append(
        "- The profiled kernel is the first `spread` launch in the benchmark binary; "
        "`assign_bins` and the reference method 3 are filtered out.\n"
        "- NCU runs with `--clock-control none` and kernel replay. Absolute durations differ "
        "from benchmark medians; utilization, instruction, and stall metrics are the useful part.\n"
        "- Source instructions and PC samples are deduplicated by `(kernel, SASS PC)`, keeping "
        "the first source correlation. Nsight repeats PCs under inline definitions and call sites.\n"
        "- Source-group percentages are a source-text heuristic, not a compiler data structure. "
        "They are meant to keep the high-level cost distribution visible across line-number drift.\n"
        "- Stall shares are `Warp State (All Cycles)` values divided by all warp-state cycles. "
        "`share of stalls` excludes the `Selected` row.\n"
        "- `raw.csv` contains every exported raw metric, including semicolon-packed lists that "
        "the detail CSV cannot expand into named columns.\n"
    )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def write_readme(out_dir: Path, order: Sequence[str]) -> None:
    lines = [
        "atomic_free_spread profiling capture",
        "===================================",
        "",
        "Regenerate everything with:",
        "",
        "    ./devel/morton_spread/profile_atomic_free_spread.py",
        "",
        "Re-analyze this directory without profiling:",
        "",
        f"    ./devel/morton_spread/profile_atomic_free_spread.py --analyze-only {out_dir}",
        "",
        "Start with summary.md. The raw and detail CSV/TXT dumps are intentionally",
        "kept so the analysis can be extended without another GPU run.",
        "",
        "Cases: " + ", ".join(order),
        "",
        "Layout:",
        "  bench/<case>.log            benchmark stdout/stderr",
        "  reports/<case>.ncu-rep      original Nsight Compute report",
        "  ncu/<case>.raw.csv          raw metric/unit/value dump",
        "  ncu/<case>.details.csv      section/metric long-form dump",
        "  ncu/<case>.details.txt      human-readable detail sections",
        "  ncu/<case>.source.csv       CUDA/SASS source counter export",
        "  ncu/<case>.source.txt       human-readable source correlation",
        "  analysis/<case>/*.csv       derived tables used by summary.md",
        "",
    ]
    (out_dir / "README.txt").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main flow


def make_out_dir(base: Optional[Path]) -> Path:
    if base is not None:
        base.mkdir(parents=True, exist_ok=True)
        return base
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPO / "build" / "gpu" / "traces" / f"af_profile_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def export_report(runner: CommandRunner, ncu: Path, report: Path, out_dir: Path, case: str) -> None:
    ncu_dir = out_dir / "ncu"
    ncu_dir.mkdir(parents=True, exist_ok=True)
    exports = [
        ("raw.csv", ["--page", "raw", "--csv"]),
        ("details.csv", ["--page", "details", "--print-details", "all", "--csv"]),
        ("details.txt", ["--page", "details", "--print-details", "all"]),
        (
            "source.csv",
            ["--page", "source", "--print-source", "cuda,sass", "--csv"],
        ),
        ("source.txt", ["--page", "source", "--print-source", "cuda,sass"]),
    ]
    for suffix, extra in exports:
        target = ncu_dir / f"{case}.{suffix}"
        err_target = ncu_dir / f"{case}.{suffix}.err"
        with target.open("w") as stdout, err_target.open("w") as stderr:
            cmd = [str(ncu), "--import", str(report), *extra]
            runner.lines.append(f"\n$ {runner.command_text(cmd)} > {target}")
            runner.flush()
            proc = subprocess.run(
                cmd,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
        if proc.returncode:
            raise RuntimeError(f"ncu export failed for {case} {suffix}")
        print(f"  exported {target.relative_to(out_dir)}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build, benchmark, profile, and summarize the atomic-free spread baseline."
    )
    parser.add_argument(
        "--cases",
        default=None,
        help=(
            "comma-separated case names (default: all cases; with "
            f"--analyze-only, the cases found in the directory; choices: {','.join(CASES)})"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: build/gpu/traces/af_profile_<timestamp>)",
    )
    parser.add_argument(
        "--analyze-only",
        type=Path,
        default=None,
        help="re-export and analyze existing reports in this directory; no build/run/profile",
    )
    parser.add_argument("--build-dir", type=Path, default=None, help="CUDA build dir")
    parser.add_argument("--cuda-home", type=Path, default=None, help="CUDA toolkit root")
    parser.add_argument("--cccl", type=Path, default=None, help="CCCL include root")
    parser.add_argument("--arch", default=None, help="nvcc arch, e.g. sm_75")
    parser.add_argument("--nrun", type=int, default=5, help="benchmark timed runs")
    parser.add_argument("--skip-build", action="store_true", help="reuse existing lineinfo binary")
    parser.add_argument("--skip-benchmark", action="store_true", help="skip plain benchmark")
    parser.add_argument("--skip-profile", action="store_true", help="skip ncu collection")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue with other cases after a failure",
    )
    args = parser.parse_args(argv)

    explicit_cases = args.cases is not None
    if explicit_cases:
        selected = [name.strip() for name in args.cases.split(",") if name.strip()]
    else:
        selected = list(CASES)
    unknown = [name for name in selected if name not in CASES]
    if unknown:
        parser.error(f"unknown case(s): {', '.join(unknown)}; choose from {', '.join(CASES)}")

    if args.analyze_only is not None:
        out_dir = args.analyze_only.resolve()
        if not out_dir.is_dir():
            parser.error(f"--analyze-only directory does not exist: {out_dir}")
        if not explicit_cases:
            available = {
                path.stem for path in (out_dir / "reports").glob("*.ncu-rep")
            }
            available |= {
                path.stem for path in (out_dir / "bench").glob("*.log")
            }
            selected = [case for case in CASES if case in available]
            if not selected:
                parser.error(
                    f"no reports or benchmark logs found under {out_dir}; "
                    "pass --cases or rerun the profiler"
                )
    else:
        out_dir = make_out_dir(args.out)

    runner = CommandRunner(out_dir / "commands.log")
    environment: Dict[str, str] = {
        "repo": str(REPO),
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": platform.python_version(),
    }
    environment["git_commit"] = runner.capture(
        ["git", "rev-parse", "HEAD"], cwd=REPO
    ).strip()
    git_status = runner.capture(["git", "status", "--short"], cwd=REPO).strip()
    environment["git_status"] = git_status or "clean"

    cuda_home = args.cuda_home or default_cuda_home()
    if cuda_home is None:
        print("error: cannot find CUDA toolkit; pass --cuda-home", file=sys.stderr)
        return 2
    cuda_home = cuda_home.resolve()
    env = build_environment(cuda_home)
    nvcc = find_executable("nvcc", [cuda_home / "bin" / "nvcc"])
    ncu = find_executable(
        "ncu",
        [
            cuda_home / "bin" / "ncu",
            Path("/opt/nvidia/nsight-compute/2025.1.0/ncu"),
            Path("/usr/local/bin/ncu"),
        ],
    )
    if nvcc is None:
        print(f"error: nvcc not found under {cuda_home}", file=sys.stderr)
        return 2
    if ncu is None and (args.analyze_only is not None or not args.skip_profile):
        print("error: ncu not found; install Nsight Compute or pass --skip-profile", file=sys.stderr)
        return 2

    build_dir = (args.build_dir or REPO / "build" / "gpu").resolve()
    cccl = args.cccl or default_cccl()
    if cccl is not None:
        cccl = cccl.resolve()
    arch = args.arch or detect_arch(find_executable("nvidia-smi", [Path("/usr/bin/nvidia-smi")]))

    environment["build_dir"] = str(build_dir)
    environment["cccl"] = str(cccl) if cccl is not None else "(not needed)"
    environment["arch"] = arch
    environment["cuda_home"] = str(cuda_home)
    environment["nvcc"] = first_line(
        runner.capture([str(nvcc), "--version"], env=env)
    )
    if ncu is not None:
        environment["ncu"] = first_line(
            runner.capture([str(ncu), "--version"], env=env)
        )
    nvidia_smi = find_executable("nvidia-smi", [Path("/usr/bin/nvidia-smi")])
    if nvidia_smi is not None:
        environment["gpu"] = first_line(
            runner.capture(
                [
                    str(nvidia_smi),
                    "--query-gpu=name,compute_cap,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                env=env,
            )
        )

    binary = build_dir / "atomic_free_spread_bench_lineinfo"
    libcufinufft = build_dir / "libcufinufft.a"
    libfinufft_common = build_dir / "src" / "common" / "libfinufft_common.a"
    poet_include = build_dir / "_deps" / "poet-src" / "include"

    if args.analyze_only is None and not args.skip_build:
        if cccl is None:
            print(
                "error: CCCL headers not found; pass --cccl (expected thrust/libcudacxx/cub)",
                file=sys.stderr,
            )
            return 2
        if not libcufinufft.is_file() or not libfinufft_common.is_file():
            print(
                f"error: missing {libcufinufft} or {libfinufft_common}.\n"
                f"Configure/build the gpu preset first, e.g. cmake --preset gpu && "
                f"cmake --build {build_dir}",
                file=sys.stderr,
            )
            return 2
        build_cmd = [
            str(nvcc),
            "-O3",
            "-std=c++17",
            f"-arch={arch}",
            "-lineinfo",
            "-Iinclude",
            f"-I{cccl / 'thrust'}",
            f"-I{cccl / 'libcudacxx' / 'include'}",
            f"-I{cccl / 'cub'}",
            f"-isystem={cuda_home / 'include'}",
            f"-isystem={poet_include}",
            "devel/morton_spread/atomic_free_spread.cu",
            "devel/morton_spread/f0_point_owned_spread.cu",
            "devel/morton_spread/atomic_free_spread_bench.cu",
            "-o",
            str(binary),
            str(libcufinufft),
            str(libfinufft_common),
            "-lcudart",
            "-lcufft",
            "-lpthread",
            "-ldl",
            "-lm",
        ]
        environment["build_command"] = runner.command_text(build_cmd)
        runner.run(build_cmd, log_name="build.log", env=env, cwd=REPO)
    else:
        environment["build_command"] = "(reused existing binary)" if binary.exists() else "(not built)"

    if (
        args.analyze_only is None
        and not binary.is_file()
        and (not args.skip_benchmark or not args.skip_profile)
    ):
        print(
            f"error: missing benchmark binary {binary}; build it first or drop --skip-build",
            file=sys.stderr,
        )
        return 2

    results: Dict[str, Dict[str, object]] = {}
    failed: List[str] = []

    for case in selected:
        print(f"\n===== {case} =====")
        case_info = CASES[case]
        case_args = [int(v) for v in case_info["args"]]  # type: ignore[arg-type]
        bench: Dict[str, object] = {}
        bench_log = out_dir / "bench" / f"{case}.log"
        report = out_dir / "reports" / f"{case}.ncu-rep"

        try:
            if args.analyze_only is None:
                if not args.skip_benchmark:
                    bench_cmd = [
                        str(binary),
                        *[str(v) for v in case_args],
                        str(args.nrun),
                    ]
                    runner.run(
                        bench_cmd,
                        log_name=f"bench/{case}.log",
                        env=env,
                        cwd=REPO,
                    )
                    bench = parse_bench_log(bench_log.read_text())
                    if bench.get("overall") not in (None, "PASS"):
                        raise RuntimeError(f"benchmark checks failed for {case}: {bench.get('overall')}")

                if not args.skip_profile:
                    if ncu is None:
                        raise RuntimeError("ncu not available")
                    report.parent.mkdir(parents=True, exist_ok=True)
                    profile_cmd = [
                        str(ncu),
                        "--clock-control",
                        "none",
                        "--kernel-name-base",
                        "function",
                        "--kernel-name",
                        "regex:^spread$",
                        "--launch-count",
                        "1",
                        "--force-overwrite",
                        *sum((["--section", section] for section in NCU_SECTIONS), []),
                        "--import-source",
                        "yes",
                        "--source-folders",
                        str(REPO),
                        "--export",
                        str(report),
                        str(binary),
                        *[str(v) for v in case_args],
                        "1",
                    ]
                    runner.run(
                        profile_cmd,
                        log_name=f"ncu/{case}.ncu.log",
                        env=env,
                        cwd=REPO,
                    )
            else:
                # Keep any benchmark log already present in the capture.
                if bench_log.is_file():
                    bench = parse_bench_log(bench_log.read_text())

            if report.is_file():
                export_report(runner, ncu, report, out_dir, case)  # type: ignore[arg-type]
                result = analyze_case(
                    case,
                    out_dir,
                    bench,
                    out_dir / "ncu" / f"{case}.details.csv",
                    out_dir / "ncu" / f"{case}.source.csv",
                )
                results[case] = result
                key = result["key_metrics"]
                print(
                    f"  {case}: bench median={fmt(bench.get('spread_median_ms'))} ms, "
                    f"ncu duration={fmt(key.get('duration_ms'))} ms, "
                    f"instructions={fmt(key.get('instructions_executed'), 0)}, "
                    f"top source={key.get('top_source_file', '?')}:{key.get('top_source_line', '?')} "
                    f"({fmt(key.get('top_source_instruction_pct'))}%)"
                )
            else:
                # Benchmark-only capture: still emit a small summary.
                if bench:
                    results[case] = {
                        "case": case,
                        "args": case_args,
                        "why": case_info["why"],
                        "bench": bench,
                        "key_metrics": {},
                        "stalls": [],
                        "source_groups": [],
                        "top_source_lines": [],
                        "top_sample_lines": [],
                        "top_opcodes": [],
                        "top_opcode_families": [],
                        "source_stats": {},
                        "artifacts": {},
                    }
                else:
                    raise RuntimeError(f"no report and no benchmark data for {case}")
        except Exception as exc:  # noqa: BLE001 - CLI wants a readable summary
            failed.append(case)
            print(f"error: {case}: {exc}", file=sys.stderr)
            if not args.keep_going:
                break

    environment["generated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    environment["failed_cases"] = ",".join(failed) if failed else ""
    (out_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    (out_dir / "environment.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in environment.items()) + "\n"
    )

    if results:
        order = [case for case in selected if case in results]
        write_summary(out_dir, environment, results, order)
        write_readme(out_dir, order)
        (out_dir / "summary.json").write_text(
            json.dumps(
                {
                    "environment": environment,
                    "cases": {case: results[case] for case in order},
                },
                indent=2,
            )
            + "\n"
        )
        # Wide cross-case table for quick spreadsheet inspection.
        wide_rows = []
        for case in order:
            key = results[case]["key_metrics"]
            bench = results[case]["bench"]
            row: Dict[str, object] = {"case": case}
            row.update({f"bench_{name}": value for name, value in bench.items()})
            row.update({f"ncu_{name}": value for name, value in key.items()})
            wide_rows.append(row)
        fieldnames = ["case"] + sorted(
            {name for row in wide_rows for name in row if name != "case"}
        )
        write_csv(out_dir / "summary_metrics.csv", fieldnames, wide_rows)
        print(f"\nWrote {out_dir / 'summary.md'}")
        print(f"Wrote {out_dir / 'summary.json'}")
        print(f"Wrote {out_dir / 'summary_metrics.csv'}")
        print(f"Raw reports: {out_dir / 'reports'}")

    if failed:
        print(f"\nFailed cases: {', '.join(failed)}", file=sys.stderr)
        return 1
    if not results:
        print("\nNo cases produced data.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
