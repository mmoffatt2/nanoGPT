#!/usr/bin/env python3
"""Utility for displaying model statistics CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Tuple, List, Iterable

from rich.console import Console
from rich.table import Table

from utils.model_stats import print_model_stats_table

STAT_KEYS = ["stdev", "kurtosis", "max", "min", "abs_max"]


def _f(val: str) -> float:
    try:
        fval = float(val)
        return fval if math.isfinite(fval) else float("nan")
    except Exception:
        return float("nan")


def load_csv(path: Path) -> Tuple[List[str], Dict[str, Dict], Dict[str, Dict]]:
    """Return row order, weight stats, and activation stats from *path*."""
    row_order: List[str] = []
    w_stats: Dict[str, Dict] = {}
    a_stats: Dict[str, Dict] = {}
    with path.open() as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # header
        for row in reader:
            name = row[0]
            row_order.append(name)
            w_vals = [_f(v) for v in row[1 : 1 + len(STAT_KEYS)]]
            a_vals = [_f(v) for v in row[1 + len(STAT_KEYS) :]]

            module = name
            if any(math.isfinite(v) for v in w_vals):
                w_stats[name] = {k: v for k, v in zip(STAT_KEYS, w_vals)}
                module = name.rsplit(".", 1)[0]

            if any(math.isfinite(v) for v in a_vals):
                a_stats[module] = {k: v for k, v in zip(STAT_KEYS, a_vals)}
    return row_order, w_stats, a_stats


def _collect_extremes(stats: Dict[str, Dict], keys: Iterable[str]) -> Dict[str, Tuple[float, float]]:
    """Return min/max extremes for each *key* over ``stats``."""
    ext: Dict[str, Tuple[float, float]] = {}
    for key in keys:
        vals: List[float] = []
        for d in stats.values():
            v = d.get(key)
            if not isinstance(v, float) or not math.isfinite(v):
                continue
            trans = -v if key != "min" else v
            vals.append(trans)
        ext[key] = (min(vals), max(vals)) if vals else (0.0, 0.0)
    return ext


def _colour_delta(val: float, key: str, lo: float, hi: float) -> str:
    if not isinstance(val, float) or not math.isfinite(val):
        return "[orange3]nan[/]"
    trans = -val if key != "min" else val
    if hi == lo:
        t = 0.5
    else:
        t = 1 - (trans - lo) / (hi - lo)
        t = max(0.0, min(1.0, t))
    r = int(255 * t)
    g = int(255 * (1 - t))
    color = f"#{r:02x}{g:02x}00"
    return f"[{color}]{val:.6f}[/]"


def print_delta(
    row_order: List[str],
    w1: Dict[str, Dict],
    a1: Dict[str, Dict],
    w2: Dict[str, Dict],
    a2: Dict[str, Dict],
    stats: Iterable[str],
) -> None:
    """Display differences between two model stats tables."""

    # compute diffs
    diff_w: Dict[str, Dict] = {}
    diff_a: Dict[str, Dict] = {}

    for name in set(w1) | set(w2):
        diff_w[name] = {}
        for k in stats:
            v1 = w1.get(name, {}).get(k, float("nan"))
            v2 = w2.get(name, {}).get(k, float("nan"))
            diff_w[name][k] = v2 - v1

    for name in set(a1) | set(a2):
        diff_a[name] = {}
        for k in stats:
            v1 = a1.get(name, {}).get(k, float("nan"))
            v2 = a2.get(name, {}).get(k, float("nan"))
            diff_a[name][k] = v2 - v1

    w_ext = _collect_extremes(diff_w, stats)
    a_ext = _collect_extremes(diff_a, stats)

    console = Console()
    table = Table(title="Δ Model Statistics", header_style="bold magenta")
    table.add_column("Tensor", no_wrap=True)
    for key in stats:
        table.add_column(f"W {key} 1", justify="right")
        table.add_column(f"W {key} 2", justify="right")
        table.add_column(f"ΔW {key}", justify="right")
    for key in stats:
        table.add_column(f"A {key} 1", justify="right")
        table.add_column(f"A {key} 2", justify="right")
        table.add_column(f"ΔA {key}", justify="right")

    # Ensure deterministic row ordering
    all_names = list(row_order)
    for n in set(w2) | set(w1):
        if n not in row_order:
            all_names.append(n)
    printed = set()

    for name in all_names:
        module = name.rsplit(".", 1)[0]
        row = [name]
        for k in stats:
            w_v1 = w1.get(name, {}).get(k, float("nan"))
            w_v2 = w2.get(name, {}).get(k, float("nan"))
            delta_w = diff_w.get(name, {}).get(k, float("nan"))
            row.append(f"{w_v1:.6f}" if math.isfinite(w_v1) else "nan")
            row.append(f"{w_v2:.6f}" if math.isfinite(w_v2) else "nan")
            row.append(_colour_delta(delta_w, k, *w_ext[k]))

        for k in stats:
            a_v1 = a1.get(module, {}).get(k, float("nan"))
            a_v2 = a2.get(module, {}).get(k, float("nan"))
            delta_a = diff_a.get(module, {}).get(k, float("nan"))
            row.append(f"{a_v1:.6f}" if math.isfinite(a_v1) else "nan")
            row.append(f"{a_v2:.6f}" if math.isfinite(a_v2) else "nan")
            row.append(_colour_delta(delta_a, k, *a_ext[k]))

        table.add_row(*row)
        printed.add(module)

    # Add activation-only modules not seen above
    for mod in diff_a.keys():
        if mod in printed:
            continue
        row = [mod]
        for k in stats:
            row.extend([
                "nan",
                "nan",
                _colour_delta(float("nan"), k, *w_ext[k]),
            ])
        for k in stats:
            a_v1 = a1.get(mod, {}).get(k, float("nan"))
            a_v2 = a2.get(mod, {}).get(k, float("nan"))
            delta_a = diff_a.get(mod, {}).get(k, float("nan"))
            row.append(f"{a_v1:.6f}" if math.isfinite(a_v1) else "nan")
            row.append(f"{a_v2:.6f}" if math.isfinite(a_v2) else "nan")
            row.append(_colour_delta(delta_a, k, *a_ext[k]))
        table.add_row(*row)

    console.print(table)


def main() -> None:
    ap = argparse.ArgumentParser(description="View model stats CSV")
    ap.add_argument("csv", nargs="+", help="One or two CSV files")
    ap.add_argument(
        "--stats",
        default="all",
        help="Comma-separated list of stats to compare (default: all)",
    )
    args = ap.parse_args()

    selected = (
        STAT_KEYS
        if args.stats.lower() == "all"
        else [s for s in args.stats.split(",") if s in STAT_KEYS]
    )

    if len(args.csv) == 1:
        _, w, a = load_csv(Path(args.csv[0]))
        print_model_stats_table(w, a)
    else:
        rows1, w1, a1 = load_csv(Path(args.csv[0]))
        rows2, w2, a2 = load_csv(Path(args.csv[1]))
        row_order = rows1
        for r in rows2:
            if r not in row_order:
                row_order.append(r)
        print_delta(row_order, w1, a1, w2, a2, selected)


if __name__ == "__main__":
    main()

