"""Tables, in the three formats this work actually needs.

``text`` for reading in a terminal, ``markdown`` for a README or an issue, and ``latex``
using booktabs for a paper. The same builder produces all three, so a number that appears
in a manuscript is the same number that appeared on screen rather than one retyped from
it, which is where transcription errors come from.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .dataset import Dataset, decode_roofline

Row = Sequence[Any]


# ------------------------------------------------------------------------ rendering

def _format_cell(value: Any, precision: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:  # NaN
            return "-"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 100:
            return f"{value:.1f}"
        return f"{value:.{precision}f}"
    return str(value)


def render(headers: Sequence[str], rows: Sequence[Row], fmt: str = "text",
           caption: Optional[str] = None, label: Optional[str] = None,
           align: Optional[str] = None) -> str:
    """Render a table in text, markdown or latex."""
    cells = [[_format_cell(v) for v in row] for row in rows]
    if fmt == "latex":
        return _render_latex(headers, cells, caption, label, align)
    if fmt == "markdown":
        return _render_markdown(headers, cells, caption)
    return _render_text(headers, cells, caption)


def _render_text(headers: Sequence[str], cells: List[List[str]],
                 caption: Optional[str]) -> str:
    all_rows = [list(headers)] + cells
    widths = [max(len(row[i]) for row in all_rows) for i in range(len(headers))]
    lines = []
    if caption:
        lines += [caption, ""]
    lines.append("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip())
    lines.append("  ".join("-" * w for w in widths))
    for row in cells:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))).rstrip())
    return "\n".join(lines)


def _render_markdown(headers: Sequence[str], cells: List[List[str]],
                     caption: Optional[str]) -> str:
    lines = []
    if caption:
        lines += [f"**{caption}**", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in cells:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _render_latex(headers: Sequence[str], cells: List[List[str]], caption: Optional[str],
                  label: Optional[str], align: Optional[str]) -> str:
    column_spec = align or ("l" + "r" * (len(headers) - 1))
    lines = ["\\begin{table}[t]", "\\centering"]
    if caption:
        lines.append("\\caption{%s}" % _latex_escape(caption))
    if label:
        lines.append("\\label{tab:%s}" % label)
    lines += ["\\begin{tabular}{%s}" % column_spec, "\\toprule"]
    lines.append(" & ".join(_latex_escape(h) for h in headers) + " \\\\")
    lines.append("\\midrule")
    for row in cells:
        lines.append(" & ".join(_latex_escape(c) for c in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _latex_escape(text: str) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                    "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(text))


# --------------------------------------------------------------------------- tables

def device_table(dataset: Dataset, fmt: str = "text") -> str:
    """The machines that produced this dataset."""
    headers = ["device", "cpu", "arch", "cores", "RAM GB", "DRAM peak GB/s", "runs"]
    rows = []
    for (device,), subset in dataset.group_by("device"):
        first = subset.rows[0]
        rows.append([
            device, first.get("cpu") or first.get("board") or "-", first.get("arch"),
            first.get("cores"),
            (first.get("ram_bytes") or 0) / 1e9 or None,
            first.get("dram_peak_GBs"), len(subset),
        ])
    return render(headers, rows, fmt, caption="Devices under test", label="devices",
                  align="lllrrrr")


def summary_table(dataset: Dataset, fmt: str = "text", limit: Optional[int] = None) -> str:
    """Every LLM throughput run, one per line."""
    # Throughput rows only: latency-mode records carry a decode rate too, but mixing the
    # two modes in one table invites comparing a batched rate to a streaming one.
    usable = (dataset.filter(mode="throughput")
              .having("decode_tps").sort("device", "weight_bytes", "threads"))
    if not usable.rows:
        return "(no throughput-mode LLM runs in this dataset)"
    headers = ["device", "model", "quant", "thr", "MB", "prefill t/s", "decode t/s",
               "GB/s", "% peak", "W", "C"]
    rows = []
    for row in (usable.rows[:limit] if limit else usable.rows):
        rows.append([
            row["device"], row["model"], row["quant"], row["threads"],
            row.get("weight_MB"), row.get("prefill_tps"), row.get("decode_tps"),
            row.get("decode_bw_GBs"), row.get("bw_util_pct"),
            row.get("power_w"), row.get("temp_c"),
        ])
    return render(headers, rows, fmt, caption="LLM decode and prefill throughput",
                  label="throughput")


def roofline_table(dataset: Dataset, fmt: str = "text") -> str:
    """The memory-wall fit, per device.

    A high R squared here is the claim that decode is bandwidth bound: throughput is
    predicted by model size alone, whatever the architecture or quantisation.
    """
    headers = ["device", "points", "effective GB/s", "R^2", "DRAM peak GB/s", "% of peak"]
    rows = []
    for (device,), subset in dataset.group_by("device"):
        fit = decode_roofline(subset)
        if fit.get("n", 0) < 2:
            continue
        rows.append([device, fit["n"], fit.get("bw_eff_GBs"), fit.get("r2"),
                     fit.get("dram_peak_GBs"), fit.get("peak_fraction_pct")])
    if not rows:
        return "(no device has enough points for a roofline fit)"
    return render(headers, rows, fmt,
                  caption="Decode roofline: tokens per second against inverse model size",
                  label="roofline")


def thread_scaling_table(dataset: Dataset, fmt: str = "text") -> str:
    """Where prefill keeps scaling and decode stops.

    The interesting number is the ratio between the two. Prefill is compute bound and
    keeps improving with threads; decode is bandwidth bound and saturates early, which is
    why the best decode thread count is often well below the core count.
    """
    usable = dataset.having("threads", "decode_tps")
    headers = ["device", "model", "threads", "prefill t/s", "decode t/s",
               "prefill speedup", "decode speedup"]
    rows = []
    for (device, model), subset in usable.group_by("device", "model"):
        ordered = subset.sort("threads")
        if len(ordered) < 2:
            continue
        base = ordered.rows[0]
        base_prefill = base.get("prefill_tps")
        base_decode = base.get("decode_tps")
        for row in ordered.rows:
            rows.append([
                device, model, row["threads"], row.get("prefill_tps"), row.get("decode_tps"),
                (row["prefill_tps"] / base_prefill)
                if row.get("prefill_tps") and base_prefill else None,
                (row["decode_tps"] / base_decode)
                if row.get("decode_tps") and base_decode else None,
            ])
    if not rows:
        return "(no model was measured at more than one thread count)"
    return render(headers, rows, fmt, caption="Thread scaling, relative to the lowest "
                  "thread count measured", label="threads")


def quantization_table(dataset: Dataset, fmt: str = "text") -> str:
    """The size and speed frontier across quantisations of the same model family."""
    usable = dataset.filter(mode="throughput").having("quant", "decode_tps",
                                                      "weight_bytes")
    headers = ["device", "model", "quant", "MB", "decode t/s", "GB/s", "mJ/token"]
    rows = []
    for (device,), subset in usable.group_by("device"):
        if len(subset.unique("quant")) < 2:
            continue
        # Best decode point per quantisation; thread variants are the thread table's job.
        best: Dict[str, Dict[str, Any]] = {}
        for row in subset.rows:
            key = f"{row['quant']}|{row.get('weight_MB')}"
            current = best.get(key)
            if current is None or row["decode_tps"] > current["decode_tps"]:
                # Prefer the variant that also measured energy when decode ties.
                best[key] = row
            elif (current and abs(row["decode_tps"] - current["decode_tps"]) < 0.5
                  and row.get("energy_per_token_mj") and not current.get("energy_per_token_mj")):
                best[key] = row
        for row in sorted(best.values(), key=lambda r: r.get("weight_MB") or 0):
            rows.append([device, row["model"], row["quant"], row.get("weight_MB"),
                         row.get("decode_tps"), row.get("decode_bw_GBs"),
                         row.get("energy_per_token_mj")])
    if not rows:
        return "(no model family was measured at more than one quantisation)"
    return render(headers, rows, fmt, caption="Quantisation frontier", label="quant")


def latency_table(dataset: Dataset, fmt: str = "text") -> str:
    """Time to first token against prompt length.

    TTFT is prefill work, so it grows with the prompt while the decode rate does not.
    That split is invisible to a throughput benchmark and is what a user actually feels.
    """
    usable = dataset.having("ttft_ms").sort("device", "model", "prompt_tokens")
    headers = ["device", "model", "prompt tok", "output tok", "TTFT ms",
               "e2e ms", "decode t/s", "implied prefill t/s"]
    rows = []
    for row in usable.rows:
        rows.append([row["device"], row["model"], row.get("prompt_tokens"),
                     row.get("output_tokens"), row.get("ttft_ms"), row.get("e2e_ms"),
                     row.get("decode_tps"), row.get("prefill_tps")])
    if not rows:
        return "(no latency-mode runs in this dataset)"
    return render(headers, rows, fmt, caption="Single-request latency", label="latency")


def energy_table(dataset: Dataset, fmt: str = "text") -> str:
    """Energy per token, and how much of it is the memory system.

    The core against DRAM split is the interesting column. When DRAM is a few percent of
    total power, the energy cost of inference is cores stalling on memory rather than the
    memory itself, which points optimisation somewhere different.
    """
    # Throughput runs only: a latency run's mean power spans prefill and idle stretches,
    # so an energy-per-token derived from it would mix two different quantities.
    usable = dataset.filter(mode="throughput").having("power_w")
    headers = ["device", "model", "quant", "thr", "W total", "W core", "W DRAM",
               "DRAM %", "mJ/token", "C"]
    rows = []
    for row in usable.sort("device", "model", "threads").rows:
        dram_pct = None
        if row.get("power_dram_w") and row.get("power_w"):
            dram_pct = 100.0 * row["power_dram_w"] / row["power_w"]
        rows.append([row["device"], row["model"], row.get("quant"), row.get("threads"),
                     row.get("power_w"), row.get("power_core_w"), row.get("power_dram_w"),
                     dram_pct, row.get("energy_per_token_mj"), row.get("temp_c")])
    if not rows:
        return "(no run in this dataset measured power)"
    return render(headers, rows, fmt, caption="Power and energy", label="energy")


def vision_table(dataset: Dataset, fmt: str = "text") -> str:
    """ONNX Runtime results, where the metric is per-inference latency, not tokens."""
    usable = dataset.having("latency_ms").sort("device", "model")
    headers = ["device", "model", "quant", "shape", "thr", "mean ms", "p50", "p95", "p99",
               "inf/s", "W"]
    rows = []
    for row in usable.rows:
        shape = row.get("input_shape")
        rows.append([row["device"], row["model"], row.get("quant"),
                     "x".join(str(d) for d in shape) if shape else "-",
                     row.get("threads"), row.get("latency_ms"), row.get("latency_p50"),
                     row.get("latency_p95"), row.get("latency_p99"), row.get("ips"),
                     row.get("power_w")])
    if not rows:
        return "(no vision runs in this dataset)"
    return render(headers, rows, fmt, caption="ONNX Runtime inference latency",
                  label="vision")


def cross_platform_table(dataset: Dataset, fmt: str = "text") -> str:
    """The same model on every device, which is the point of a hardware abstraction."""
    devices = dataset.unique("device")
    if len(devices) < 2:
        return "(only one device in this dataset; nothing to compare)"

    usable = dataset.having("decode_tps")
    headers = ["model", "quant"] + [f"{d} t/s" for d in devices] + ["ratio"]
    rows = []
    for (model, quant), subset in usable.group_by("model", "quant"):
        per_device = {}
        for (device,), device_subset in subset.group_by("device"):
            best = device_subset.best("decode_tps")
            if best:
                per_device[device] = best["decode_tps"]
        if len(per_device) < 2:
            continue
        values = [per_device.get(d) for d in devices]
        present = [v for v in values if v]
        ratio = max(present) / min(present) if len(present) > 1 else None
        rows.append([model, quant] + values + [ratio])
    if not rows:
        return "(no model was measured on more than one device)"
    return render(headers, rows, fmt,
                  caption="Same model, every device: best decode rate", label="crossplat")


ALL_TABLES: List[Tuple[str, Callable[..., str]]] = [
    ("Devices", device_table),
    ("Throughput", summary_table),
    ("Decode roofline", roofline_table),
    ("Thread scaling", thread_scaling_table),
    ("Quantisation", quantization_table),
    ("Latency", latency_table),
    ("Energy", energy_table),
    ("Vision", vision_table),
    ("Cross-platform", cross_platform_table),
]
