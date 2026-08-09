"""Analysis layer: a results directory in, tables and figures out.

The one place in the package where numpy and matplotlib are allowed. Everything below the
measurement line is standard library so it can run on the device under test; everything
here runs on the analyst's machine.
"""

from __future__ import annotations

import os
from typing import List, Optional

from ..schema import load_records
from .dataset import Dataset, decode_roofline, linear_fit_through_origin
from . import tables


def report(results_dir: str, output_dir: Optional[str] = None,
           formats: tuple = ("png", "pdf"), table_format: str = "markdown") -> str:
    """The one-command path: results directory in, REPORT.md plus figures out."""
    dataset = Dataset.from_directory(results_dir)
    if not len(dataset):
        raise ValueError(f"no successful run records under {results_dir}")

    output_dir = output_dir or os.path.join(results_dir, "report")
    os.makedirs(output_dir, exist_ok=True)

    figures: List[str] = []
    try:
        from .plots import FigureSet

        figure_set = FigureSet(dataset, output_dir, formats=formats)
        figures = figure_set.generate_all()
    except ImportError:
        pass  # tables alone are still a report

    lines: List[str] = ["# Benchmark report", ""]

    warnings = dataset.integrity_warnings()
    if warnings:
        lines.append("> **Integrity notes**")
        for note in warnings:
            lines.append(f"> - {note}")
        lines.append("")

    for title, builder in tables.ALL_TABLES:
        rendered = builder(dataset, fmt=table_format)
        if rendered.startswith("("):
            continue  # nothing measured for this table; omit rather than apologise
        lines += [f"## {title}", "", rendered, ""]

    if figures:
        lines.append("## Figures")
        lines.append("")
        for figure in figures:
            lines.append(f"![{figure}]({figure}.png)")
        lines.append("")

    report_path = os.path.join(output_dir, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return report_path


__all__ = ["Dataset", "decode_roofline", "linear_fit_through_origin", "report",
           "tables", "load_records"]
