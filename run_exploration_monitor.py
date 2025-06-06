# run_exploration_monitor.py
"""
Textual app to monitor hyperparameter search results from a YAML log file.
Refreshes every N seconds, showing all runs in a DataTable view.
Interactive keybindings:
  Enter - toggle sort by column
  h/l   - move column left/right
  d     - hide column
  o     - unhide all columns
  x     - hide all rows matching current cell in column
  i     - invert filter: keep only rows matching current cell in column
  O     - unhide all rows (clear row filters)
  v     - export current view to CSV
  s     - save current layout
  p     - shows help menu
  g     - graphs first two rows
  L     - graph & connect points sharing the 3rd column value
  1–9   - graph & connect points sharing merged columns 3..(2+N)
  q # # - multibarcharts - `q [1-9] [1-9]` - e.g. 'q 3 2' will create bar charts for columns 1 2 and 3, the next two columns (column 4 and column 5) as merged labels.
  w–y   - barcharts with labels merged (w=1, y =5)
  c     - toggle colour-map on first column (green → red)
  u     - unsort / remove current column from the sort stack
  U     - clear *all* sorting

Use `--hotkeys` to print this help and exit.
"""

import plot_view
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import math
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import DataTable, Footer, Header


def load_runs(log_file: Path) -> List[Dict]:
    """
    Load all YAML documents from the given log file into a list.
    """
    docs: List[Dict] = []
    if not log_file.exists():
        return docs
    with log_file.open() as f:
        for doc in yaml.safe_load_all(f):
            if doc:
                docs.append(doc)
    return docs


HOTKEYS_TEXT = (
    "Enter: toggle sort by column\n"
    "h/l: move column left/right\n"
    "d: hide column\n"
    "o: unhide all columns\n"
    "x: hide rows matching value\n"
    "i: keep only rows matching value\n"
    "O: clear all row filters\n"
    "e: export CSV\n"
    "s: save layout (columns, hidden-cols, filters)\n"
    "g: graph first two columns (matplotlib)\n"
    "g: graph first two columns (opens a Plotly window)\n"
    "p: shows help menu\n"
    "L: graph & connect points sharing the 3rd column value\n"
    "1–9: graph & connect points sharing merged columns 3..(2+N)\n"
    "q # #: multibarcharts - `q [1-9] [1-9]` - e.g. 'q 3 2' will create bar charts for columns 1 2 and 3, the next two columns (column 4 and column 5) as merged labels\n"
    "w–y: barcharts with labels merged (w=1, y =5)\n"
    "c: toggle colour-map on first column (green → red)\n"
    "u: unsort / remove current column from the sort stack\n"
    "U: clear *all* sorting\n"
)


class MonitorApp(App):
    CSS = """
    Screen { align: center middle; }
    Container { height: 1fr; }
    DataTable#table {
        height: 1fr;
        width: 1fr;
        overflow-x: auto;
        overflow-y: auto;
    }
    """

    def __init__(self, log_file: Path, interval: float) -> None:
        super().__init__()
        self.log_file = log_file
        self.interval = interval
        # Use JSON config file with same base name as YAML log file
        self.config_file = log_file.parent / f"{log_file.name}_monitor.json"
        self.param_keys: List[str] = []
        self.columns: List[str] = []
        self.all_columns: List[str] = []
        self.hidden_cols: set[str] = set()
        self.sort_stack: List[tuple[int, bool]] = []
        self.table: Optional[DataTable] = None
        self.original_entries: List[Dict] = []  # Unfiltered data
        self.current_entries: List[Dict] = []  # View data with filters
        self.row_filters: List[tuple] = []     # (col, op, val) triples
        self.colour_first: bool = False        # toggled with “c”
        self._bar_mode: bool = False           # are we collecting digits?
        self._bar_digits: List[int] = []       # collected numeric keys

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            DataTable(id="table", zebra_stripes=True),
        )
        yield Footer()

    def on_mount(self) -> None:
        self.table = self.query_one(DataTable)
        # Load YAML runs
        self.original_entries = load_runs(self.log_file)
        self.current_entries = list(self.original_entries)
        # Determine parameter keys across all runs
        keys = set()
        for entry in self.original_entries:
            keys.update(entry.get("config", {}).keys())
        self.param_keys = sorted(keys)
        # Base columns: metrics + parameters
        base_cols = ["best_val_loss", "best_val_iter", "num_params", "peak_gpu_mb", "iter_latency_avg"] + self.param_keys
        self.all_columns = base_cols.copy()
        self.columns = base_cols.copy()
        # Load persisted layout if exists
        if self.config_file.exists():
            cfg = json.loads(self.config_file.read_text())
            self.all_columns = cfg.get("all_columns", self.all_columns)
            self.hidden_cols = set(cfg.get("hidden_cols", []))
            self.columns = [c for c in self.all_columns if c not in self.hidden_cols]
            self.sort_stack = [tuple(p) for p in cfg.get("sort_stack", [])]
            # Restore saved row filters
            self.row_filters = cfg.get("row_filters", [])
            self.current_entries = list(self.original_entries)
            for col, op, val in self.row_filters:
                if op == "hide":
                    self.current_entries = [
                        e for e in self.current_entries if str(self.get_cell(e, col)) != val
                    ]
                elif op == "keep":
                    self.current_entries = [
                        e for e in self.current_entries if str(self.get_cell(e, col)) == val
                    ]
        # Build table and start refresh loop
        self.build_table()
        # Periodically invoke `refresh_table`; the DataTable mutations themselves
        # trigger a screen repaint, so no extra flag is needed.
        self.set_interval(self.interval, self.refresh_table)
        self.refresh_table()

    def build_table(self) -> None:
        """
        Clear and (re)build columns in the DataTable.
        """
        if not self.table:
            return
        self.table.clear(columns=True)
        for col in self.columns:
            self.table.add_column(col, width=max(12, len(col) + 2))

    def get_cell(self, entry: Dict, col_name: str):
        """Retrieve the value for a given column in an entry."""
        if col_name in ("best_val_loss", "best_val_iter", "num_params", "peak_gpu_mb", "iter_latency_avg"):
            return entry.get(col_name)
        return entry.get("config", {}).get(col_name)

    @staticmethod
    def _is_missing(v) -> bool:
        """Return *True* for values that should always sink to the bottom."""
        return (
            v is None
            or (isinstance(v, float) and math.isnan(v))
        )
    @classmethod
    def _sort_key(cls, v):
        """
        Build a key that always puts “missing” values (*None* or *NaN*) LAST,
        regardless of ascending / descending order; for everything else use the
        raw value when comparable, falling back to its string representation for
        mixed types.
        """
        if cls._is_missing(v):
            return (1, "")            #  sink
        return (0, v if isinstance(v, (int, float, str)) else str(v))

    def apply_progressive_sort(self) -> None:
        """
        Stable multi-key sort:
        later keys in ``self.sort_stack`` take precedence, but the ordering
        within equal keys respects the earlier sorts (Python sort is stable).
        """
        # NOTE:  we iterate **in insertion order** (oldest → newest).  
        # Because Python’s sort is *stable*, the **last** pass has the
        # highest precedence — therefore the *most-recent* “Enter” press
        # wins, exactly as requested.
        for col_idx, asc in self.sort_stack:
            col_name = self.columns[col_idx]
            self.current_entries.sort(
                key=lambda e: self._sort_key(self.get_cell(e, col_name)),
                reverse=not asc,
            )
            # second pass → ensure None/NaN always sink
            self.current_entries.sort(
                key=lambda e: self._is_missing(self.get_cell(e, col_name))
            )

    @staticmethod
    def _compare_values(a, b) -> int:
        """
        Compare *a* and *b* so that **None is always treated as the lowest-priority
        value**, i.e. it is pushed to the bottom of the table no matter the
        sort direction.

        Returns:
            -1 if a < b, 0 if a == b, 1 if a > b (w.r.t. this custom order).
        """
        if a is None and b is None:
            return 0
        if a is None:
            return 1          # a goes after b
        if b is None:
            return -1         # a goes before b
        try:
            if a < b:
                return -1
            if a > b:
                return 1
            return 0
        except TypeError:
            # Fall back to string comparison when types are incomparable (e.g., int vs str)
            sa, sb = str(a), str(b)
            if sa < sb:
                return -1
            if sa > sb:
                return 1
            return 0


    def refresh_table(self, new_cursor: Optional[int] = None) -> None:
        """Reload data, apply sorting, and repopulate the DataTable."""
        if not self.table:
            return
        # Always reload the YAML log file so new runs appear
        new_original = load_runs(self.log_file)
        if new_original != self.original_entries:
            self.original_entries = new_original

        # Re-apply any active row filters
        base_entries = list(self.original_entries)
        for col, op, val in self.row_filters:
            if op == "hide":
                base_entries = [e for e in base_entries if str(self.get_cell(e, col)) != val]
            elif op == "keep":
                base_entries = [e for e in base_entries if str(self.get_cell(e, col)) == val]
        self.current_entries = base_entries
        # Apply sort
        if self.sort_stack:
            self.apply_progressive_sort()
        # Save cursor position
        old = self.table.cursor_coordinate
        ri = old.row if old else 0
        ci = new_cursor if new_cursor is not None else (old.column if old else 0)

        # ── pre-compute colour-map for first column if enabled ─────────────
        colour_map: List[str | None] = []
        if self.colour_first and self.current_entries and self.columns:
            first_col = self.columns[0]
            numeric_vals = [
                self.get_cell(e, first_col)
                for e in self.current_entries
                if isinstance(self.get_cell(e, first_col), (int, float))
            ]
            if numeric_vals:
                lo, hi = min(numeric_vals), max(numeric_vals)
                if hi == lo:                      # avoid ÷0
                    hi += 1e-9
                rng = hi - lo
                for e in self.current_entries:
                    v = self.get_cell(e, first_col)
                    if isinstance(v, (int, float)):
                        t = (v - lo) / rng          # 0→green, 1→red
                        r = int(255 * t)
                        g = int(255 * (1 - t))
                        colour_map.append(f"#{r:02x}{g:02x}00")
                    else:
                        colour_map.append(None)
            else:
                colour_map = [None] * len(self.current_entries)
        else:
            colour_map = [None] * len(self.current_entries)

        # Rebuild columns and rows
        self.build_table()
        for row_idx, entry in enumerate(self.current_entries):
            row: List[str] = []
            for col in self.columns:
                val = self.get_cell(entry, col)
                if col == "best_val_loss" and val is not None:
                    row.append(f"{val:.6f}")
                else:
                    row.append(str(val))
            # apply colour markup to first cell if requested
            if colour_map[row_idx]:
                row[0] = f"[{colour_map[row_idx]}]{row[0]}[/]"
            self.table.add_row(*row)
        # Restore cursor
        maxr, maxc = len(self.current_entries) - 1, len(self.columns) - 1
        self.table.cursor_coordinate = (min(max(ri, 0), maxr), min(max(ci, 0), maxc))

    def _msg(self, text: str, timeout: float = 2.0) -> None:
        """
        Show a transient 2-second notification at the bottom-right.
        Works on Textual ≥0.32; falls back to a console bell if unavailable.
        """
        try:
            # Textual's builtin notifier (dismisses automatically)
            self.notify(text, timeout=timeout)
        except AttributeError:
            # Older Textual: just beep so the user at least gets feedback
            self.bell()

    async def on_key(self, event: events.Key) -> None:
        """Handle key presses for table interactions and config saving."""
        if not self.table:
            return
        coord = self.table.cursor_coordinate
        if not coord:
            return
        r, c = coord.row, coord.column
        key = event.key
        if self._bar_mode:
            if key.isdigit() and key != "0":
                self._bar_digits.append(int(key))
                if len(self._bar_digits) == 2:
                    n_bars, n_labels = self._bar_digits
                    # reset mode before plotting so errors don’t trap us
                    self._bar_mode, self._bar_digits = False, []
                    try:
                        needed = 1 + n_bars + n_labels
                        if len(self.columns) < needed:
                            raise ValueError(
                                f"Need at least {needed} visible columns "
                                "for this bar-chart"
                            )
                        y_cols   = self.columns[0 : n_bars]
                        lbl_cols = self.columns[n_bars : n_bars + n_labels]
                        plot_view.plot_multi_bars(
                            self.current_entries,
                            y_cols   = y_cols,
                            label_cols = lbl_cols,
                        )
                        self._msg(
                            f"Bar-chart: {', '.join(y_cols)} by "
                            f"{', '.join(lbl_cols)}",
                            timeout = 3,
                        )
                    except Exception as exc:
                        self._msg(f"Graph error: {exc}", timeout = 4)
            else:
                # any non-digit cancels the mode
                self._bar_mode, self._bar_digits = False, []
                self._msg("Bar-chart mode cancelled")
            return  # swallow key while in bar-mode

        # ────────────────────────── normal hotkeys ────────────────────────────

        if key == "q":
            # enter modal bar-chart mode
            self._bar_mode, self._bar_digits = True, []
            self._msg("Bar-chart mode: type <#metrics><#labels>")
            return

        if key == "e":
            # Export CSV
            fname = f"{self.log_file.stem}_export_{int(time.time())}.csv"
            with open(fname, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(self.columns)
                for entry in self.current_entries:
                    row: List[str] = []
                    for col in self.columns:
                        val = self.get_cell(entry, col)
                        if col == "best_val_loss" and val is not None:
                            row.append(f"{val:.6f}")
                        else:
                            row.append(str(val))
                    w.writerow(row)
            self.bell()
            self._msg(f"Exported view → {fname}")
        elif key == "s":
            # Save layout to JSON with same base name
            cfg = {
                "all_columns": self.all_columns,
                "hidden_cols": list(self.hidden_cols),
                "sort_stack":  [[i, asc] for i, asc in self.sort_stack],
                "row_filters": getattr(self, "row_filters", []),
            }
            self.config_file.write_text(json.dumps(cfg, indent=2))
            self.bell()
            self._msg("Layout saved")
        elif key == "enter":
            # ── direction-cycling progressive sort ──────────────────
            idx = next((k for k, (col, _) in enumerate(self.sort_stack) if col == c), None)

            if idx is None:                      # not yet in stack → ascending
                self.sort_stack.append((c, True))
                state_txt = "↑"
            else:
                col, asc = self.sort_stack.pop(idx)        # remove old entry
                if asc:                     # ascend → descend
                    self.sort_stack.append((col, False))
                    state_txt = "↓"
                else:                       # descend → remove
                    state_txt = "✕"         # no sort for this col
            self.refresh_table(new_cursor=c)

            if self.sort_stack:
                parts = [
                    f"{self.columns[i]}{'↑' if asc else '↓'}"
                    for i, asc in self.sort_stack
                ]
                self._msg(" • ".join(parts))
            else:
                self._msg("Sorting cleared")
        elif key in ("h", "l"):
            # Move column
            t = c - 1 if key == "h" else c + 1
            if 0 <= t < len(self.columns):
                n1, n2 = self.columns[c], self.columns[t]
                i1, i2 = self.all_columns.index(n1), self.all_columns.index(n2)
                self.all_columns[i1], self.all_columns[i2] = (
                    self.all_columns[i2],
                    self.all_columns[i1],
                )
                self.columns = [
                    col for col in self.all_columns if col not in self.hidden_cols
                ]
                self.refresh_table(new_cursor=t)
        elif key == "d":
            # Hide column
            col = self.columns[c]
            self.hidden_cols.add(col)
            self.columns = [
                col for col in self.all_columns if col not in self.hidden_cols
            ]
            self.refresh_table(new_cursor=c)
        elif key == "o":
            # Unhide all columns
            self.hidden_cols.clear()
            self.columns = self.all_columns.copy()
            self.refresh_table(new_cursor=c)
        elif key == "x":
            # Hide matching rows
            col = self.columns[c]
            val = str(self.get_cell(self.current_entries[r], col))
            self.current_entries = [
                e for e in self.current_entries if str(self.get_cell(e, col)) != val
            ]
            self.row_filters = self.row_filters + [(col, "hide", val)]
            self.refresh_table(new_cursor=r)
        elif key == "i":
            # Inverse filter (keep only matching)
            col = self.columns[c]
            val = str(self.get_cell(self.current_entries[r], col))
            self.current_entries = [
                e for e in self.current_entries if str(self.get_cell(e, col)) == val
            ]
            self.row_filters = self.row_filters + [(col, "keep", val)]
            self.refresh_table(new_cursor=0)
        elif key == "O":
            # Reset row filters
            self.current_entries, self.row_filters = list(self.original_entries), []
            self.refresh_table(new_cursor=0)
        elif key == "p":
            self._msg(HOTKEYS_TEXT, timeout=10.0)
        elif key == "g":
            # ── Graph using first two visible columns: col[0] ⇒ Y, col[1] ⇒ X ──
            try:
                if len(self.columns) < 2:
                    raise ValueError("Need at least two visible columns to graph")
                y_col, x_col = self.columns[0], self.columns[1]
                plot_view.plot_rows(self.current_entries, x=x_col, y=y_col)
                self._msg(f"Plotted {y_col} vs {x_col}", timeout=3)
            except Exception as exc:
                self._msg(f"Graph error: {exc}", timeout=4)
        elif key == "L":
            try:
                if len(self.columns) < 3:
                    raise ValueError("Need at least three visible columns for 'L'")
                y_col, x_col, grp_col = self.columns[0], self.columns[1], self.columns[2]
                plot_view.plot_rows(
                    self.current_entries, x=x_col, y=y_col, connect_by=grp_col
                )
                self._msg(f"Plotted {y_col} vs {x_col} grouped by {grp_col}", timeout=3)
            except Exception as exc:
                self._msg(f"Graph error: {exc}", timeout=4)
        elif key == "c":
            self.colour_first = not self.colour_first
            state = "enabled" if self.colour_first else "disabled"
            self.refresh_table()
            self._msg(f"First-column colour-map {state}")
        elif key == "u":
            # ── remove current column from sort stack ────────────────────
            before = len(self.sort_stack)
            self.sort_stack = [(col, asc) for col, asc in self.sort_stack if col != c]
            if len(self.sort_stack) != before:
                self.refresh_table(new_cursor=c)
                if self.sort_stack:
                    parts = [
                        f"{self.columns[i]}{'↑' if asc else '↓'}"
                        for i, asc in self.sort_stack
                    ]
                    self._msg("Sort order: " + " • ".join(parts))
                else:
                    self._msg("Sorting cleared")
            else:
                self._msg("Column wasn’t in sort order")
        elif key == "U":
            if self.sort_stack:
                self.sort_stack.clear()
                self.refresh_table(new_cursor=c)
                self._msg("All sorting cleared")
            else:
                self._msg("No active sorting")
        elif key.isdigit() and key != "0":  # keys '1'–'9'
            n = int(key)
            try:
                needed = 2 + n          # 0-based → need at least 2+n columns
                if len(self.columns) < needed:
                    raise ValueError(
                        f"Need at least {needed} visible columns for '{key}'"
                    )
                y_col, x_col = self.columns[0], self.columns[1]
                merge_cols   = self.columns[2 : 2 + n]

                merged_rows = []
                for e in self.current_entries:
                    parts = [str(self.get_cell(e, col)) for col in merge_cols]
                    merge_key = "-".join(parts)
                    merged_rows.append({**e, "__merge__": merge_key})

                plot_view.plot_rows(
                    merged_rows,
                    x=x_col,
                    y=y_col,
                    connect_by="__merge__",
                    connect_label="-".join(merge_cols),   # NEW
                )
                self._msg(
                    f"Plotted {y_col} vs {x_col} grouped by {'-'.join(merge_cols)}",
                    timeout=3,
                )
            except Exception as exc:
                self._msg(f"Graph error: {exc}", timeout=4)

        shift_map = {
            "w": 1,
            "e": 2,
            "r": 3,
            "t": 4,
            "y": 5,
        }
        if key in shift_map:
            n = shift_map[key]
            try:
                needed = 1 + n             # leftmost + n label columns
                if len(self.columns) < needed:
                    raise ValueError(f"Need at least {needed} visible columns")

                y_col = self.columns[0]
                lbl_cols = self.columns[1 : 1 + n]

                plot_view.plot_bars(
                    self.current_entries,
                    y=y_col,
                    label_cols=lbl_cols,
                )
                self._msg(
                    f"Bar-chart of {y_col} by {'-'.join(lbl_cols)}",
                    timeout=3,
                )
            except Exception as exc:
                self._msg(f"Graph error: {exc}", timeout=4)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor hyperparameter search results (Textual TUI)."
    )
    parser.add_argument("log_file", type=Path, help="Path to YAML log file")
    parser.add_argument(
        "--interval", type=float, default=30.0, help="Refresh interval seconds"
    )
    parser.add_argument(
        "--hotkeys", action="store_true", help="Print available hotkeys and exit"
    )
    args = parser.parse_args()

    if args.hotkeys:
        print(HOTKEYS_TEXT)
        sys.exit(0)

    app = MonitorApp(args.log_file, args.interval)
    app.run()


if __name__ == "__main__":
    main()

