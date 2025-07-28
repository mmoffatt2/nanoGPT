#!/usr/bin/env python3
"""Utility for displaying model statistics CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Tuple, List

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


def load_csv(path: Path) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Return weight and activation stats from *path*."""
    w_stats: Dict[str, Dict] = {}
    a_stats: Dict[str, Dict] = {}
    with path.open() as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # header
        for row in reader:
            name = row[0]
            w_vals = [_f(v) for v in row[1 : 1 + len(STAT_KEYS)]]
            a_vals = [_f(v) for v in row[1 + len(STAT_KEYS) :]]

            module = name
            if any(math.isfinite(v) for v in w_vals):
                w_stats[name] = {k: v for k, v in zip(STAT_KEYS, w_vals)}
                module = name.rsplit(".", 1)[0]

            if any(math.isfinite(v) for v in a_vals):
                a_stats[module] = {k: v for k, v in zip(STAT_KEYS, a_vals)}
    return w_stats, a_stats


def _collect_extremes(diff_w: Dict[str, Dict], diff_a: Dict[str, Dict]) -> Dict[str, Tuple[float, float]]:
    ext: Dict[str, Tuple[float, float]] = {}
    for key in STAT_KEYS:
        vals: List[float] = []
        for d in list(diff_w.values()) + list(diff_a.values()):
            v = d.get(key)
            if not isinstance(v, float) or not math.isfinite(v):
                continue
            trans = -v if key != "min" else v
            vals.append(trans)
        if vals:
            ext[key] = (min(vals), max(vals))
        else:
            ext[key] = (0.0, 0.0)
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


def print_delta(w1: Dict[str, Dict], a1: Dict[str, Dict], w2: Dict[str, Dict], a2: Dict[str, Dict]) -> None:
    diff_w: Dict[str, Dict] = {}
    diff_a: Dict[str, Dict] = {}

    for name in set(w1) | set(w2):
        diff_w[name] = {}
        for k in STAT_KEYS:
            v1 = w1.get(name, {}).get(k, float("nan"))
            v2 = w2.get(name, {}).get(k, float("nan"))
            diff_w[name][k] = v2 - v1

    for name in set(a1) | set(a2):
        diff_a[name] = {}
        for k in STAT_KEYS:
            v1 = a1.get(name, {}).get(k, float("nan"))
            v2 = a2.get(name, {}).get(k, float("nan"))
            diff_a[name][k] = v2 - v1

    ext = _collect_extremes(diff_w, diff_a)

    console = Console()
    table = Table(title="Δ Model Statistics", header_style="bold magenta")
    table.add_column("Tensor", no_wrap=True)
    for key in STAT_KEYS:
        table.add_column(f"W {key}", justify="right")
        table.add_column(f"A {key}", justify="right")

    printed = set()
    for name in set(w2) | set(w1):
        module = name.rsplit(".", 1)[0]
        row = [name]
        dw = diff_w.get(name, {})
        da = diff_a.get(module, {})
        for k in STAT_KEYS:
            row.append(_colour_delta(dw.get(k, float("nan")), k, *ext[k]))
            row.append(_colour_delta(da.get(k, float("nan")), k, *ext[k]))
        table.add_row(*row)
        printed.add(module)

    for mod in diff_a.keys():
        if mod in printed:
            continue
        row = [mod]
        for k in STAT_KEYS:
            row.append(_colour_delta(float("nan"), k, *ext[k]))
            row.append(_colour_delta(diff_a[mod].get(k, float("nan")), k, *ext[k]))
        table.add_row(*row)

    console.print(table)


def main() -> None:
    ap = argparse.ArgumentParser(description="View model stats CSV")
    ap.add_argument("csv", nargs="+", help="One or two CSV files")
    args = ap.parse_args()

    if len(args.csv) == 1:
        w, a = load_csv(Path(args.csv[0]))
        print_model_stats_table(w, a)
    else:
        w1, a1 = load_csv(Path(args.csv[0]))
        w2, a2 = load_csv(Path(args.csv[1]))
        print_delta(w1, a1, w2, a2)


if __name__ == "__main__":
    main()

