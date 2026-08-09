"""Publication-quality figures from a dataset. Requires matplotlib.

Figure conventions, applied consistently rather than per plot:

* **Color follows the entity.** A device keeps its color across every figure in a
  report, assigned in a fixed categorical order, never cycled. A filtered dataset must
  not repaint the survivors.
* **Every series also gets a marker shape.** The figures end up in printed papers, and
  grayscale is the one rendering the palette cannot survive alone.
* **One axis per figure.** Two measures of different scale become two figures.
* **Direct labels over legend boxes where the geometry allows.** A roofline with two
  devices needs two labels next to two lines, not a box occluding the fit.

The palette is the validated reference categorical set (light surface, print-safe
order); marker shapes carry identity when color cannot.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .dataset import Dataset, decode_roofline, linear_fit_through_origin

# Validated categorical order for a light (paper) surface. Identity is assigned by
# first appearance and held for the lifetime of a FigureSet, never re-derived per plot.
_SERIES_HEX = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
_MARKERS = ["o", "s", "^", "D", "v", "P"]

_INK = "#0b0b0b"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise ImportError(
            "plotting needs matplotlib (pip install matplotlib)"
        ) from exc


def _style(plt) -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "font.family": "sans-serif",
        "axes.edgecolor": _BASELINE, "axes.labelcolor": _INK,
        "axes.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": _GRID, "grid.linewidth": 0.6,
        "xtick.color": _MUTED, "ytick.color": _MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 6,
    })


class FigureSet:
    """Generates the standard figures for a results directory.

    Series identity (device, usually) is bound to a color and marker on first sight and
    held stable across every figure the set produces.
    """

    def __init__(self, dataset: Dataset, output_dir: str, formats: Sequence[str] = ("png", "pdf")):
        self.dataset = dataset
        self.output_dir = output_dir
        self.formats = list(formats)
        self._identity: Dict[str, int] = {}
        self.written: List[str] = []

    # -------------------------------------------------------------------- identity

    def _slot(self, key: str) -> int:
        if key not in self._identity:
            self._identity[key] = len(self._identity) % len(_SERIES_HEX)
        return self._identity[key]

    def color(self, key: str) -> str:
        return _SERIES_HEX[self._slot(key)]

    def marker(self, key: str) -> str:
        return _MARKERS[self._slot(key)]

    # --------------------------------------------------------------------- output

    def _save(self, plt, figure, name: str) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        for fmt in self.formats:
            path = os.path.join(self.output_dir, f"{name}.{fmt}")
            figure.savefig(path)
            self.written.append(path)
        plt.close(figure)

    # -------------------------------------------------------------------- figures

    def roofline(self, name: str = "fig_roofline") -> Optional[str]:
        """Decode tok/s against model bytes, log-log, with the per-device fit.

        The signature of the memory wall: every model on a device lies on one line of
        slope -1, and the line's height is the device's effective bandwidth.
        """
        plt = _require_matplotlib()
        _style(plt)
        usable = self.dataset.having("decode_tps", "weight_bytes")
        if len(usable) < 2:
            return None

        figure, axis = plt.subplots(figsize=(4.2, 3.2))
        for (device,), subset in usable.group_by("device"):
            # Best thread count per model: the roofline is about the ceiling, and
            # sub-saturated points would smear a thread effect into a size effect.
            best_rows: Dict[str, Dict[str, Any]] = {}
            for row in subset.rows:
                current = best_rows.get(row["model"])
                if current is None or row["decode_tps"] > current["decode_tps"]:
                    best_rows[row["model"]] = row
            rows = sorted(best_rows.values(), key=lambda r: r["weight_bytes"])
            sizes = [r["weight_bytes"] / 1e6 for r in rows]
            rates = [r["decode_tps"] for r in rows]
            fit = linear_fit_through_origin(
                [1.0 / r["weight_bytes"] for r in rows], rates
            )
            color, marker = self.color(device), self.marker(device)
            axis.plot(sizes, rates, linestyle="none", marker=marker, color=color,
                      markeredgecolor="white", markeredgewidth=0.6, zorder=3)
            if not math.isnan(fit["slope"]):
                span = [min(sizes) * 0.8, max(sizes) * 1.25]
                line = [fit["slope"] / (s * 1e6) for s in span]
                axis.plot(span, line, color=color, linewidth=1.2, alpha=0.75, zorder=2)
                label = f"{device}: {fit['slope'] / 1e9:.1f} GB/s"
                if not math.isnan(fit["r2"]):
                    label += f", $R^2$={fit['r2']:.3f}"
                axis.annotate(label, xy=(span[0], line[0]),
                              xytext=(4, 4), textcoords="offset points",
                              fontsize=8, color=color)

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("model weights (MB)")
        axis.set_ylabel("decode throughput (tok/s)")
        self._save(plt, figure, name)
        return name

    def thread_scaling(self, name: str = "fig_threads") -> Optional[str]:
        """Prefill and decode against threads: one saturates, the other does not.

        Same measure (relative speedup) on one axis, so prefill and decode can share it
        honestly; the two are distinguished by line style, the device by color.
        """
        plt = _require_matplotlib()
        _style(plt)
        usable = self.dataset.having("threads", "decode_tps", "prefill_tps")
        groups = [(key, subset) for key, subset in usable.group_by("device", "model")
                  if len(subset.unique("threads")) >= 2]
        if not groups:
            return None

        figure, axis = plt.subplots(figsize=(4.2, 3.2))
        seen_devices = []
        for (device, model), subset in groups:
            ordered = subset.sort("threads")
            threads = [r["threads"] for r in ordered.rows]
            base_prefill = ordered.rows[0]["prefill_tps"]
            base_decode = ordered.rows[0]["decode_tps"]
            color = self.color(device)
            axis.plot(threads, [r["prefill_tps"] / base_prefill for r in ordered.rows],
                      color=color, marker=self.marker(device), markersize=4,
                      markeredgecolor="white", markeredgewidth=0.5)
            axis.plot(threads, [r["decode_tps"] / base_decode for r in ordered.rows],
                      color=color, marker=self.marker(device), markersize=4,
                      linestyle="--", alpha=0.8,
                      markeredgecolor="white", markeredgewidth=0.5)
            if device not in seen_devices:
                seen_devices.append(device)

        axis.plot([], [], color=_INK, label="prefill")
        axis.plot([], [], color=_INK, linestyle="--", label="decode")
        for device in seen_devices:
            axis.plot([], [], color=self.color(device), marker=self.marker(device),
                      linestyle="none", label=device)
        axis.legend(loc="upper left")
        axis.set_xlabel("threads")
        axis.set_ylabel("speedup vs. 1 thread")
        self._save(plt, figure, name)
        return name

    def ttft(self, name: str = "fig_ttft") -> Optional[str]:
        """Time to first token against prompt length, per device and model."""
        plt = _require_matplotlib()
        _style(plt)
        usable = self.dataset.having("ttft_ms", "prompt_tokens")
        if len(usable) < 2:
            return None

        figure, axis = plt.subplots(figsize=(4.2, 3.2))
        labelled = False
        for (device, model), subset in usable.group_by("device", "model"):
            ordered = subset.sort("prompt_tokens")
            if len(ordered) < 2:
                continue
            prompts = [r["prompt_tokens"] for r in ordered.rows]
            ttfts = [r["ttft_ms"] for r in ordered.rows]
            color = self.color(device)
            axis.plot(prompts, ttfts, color=color, marker=self.marker(device),
                      markeredgecolor="white", markeredgewidth=0.6)
            axis.annotate(f"{device} {model}", xy=(prompts[-1], ttfts[-1]),
                          xytext=(5, 0), textcoords="offset points",
                          fontsize=8, color=color, va="center")
            labelled = True
        if not labelled:
            plt.close(figure)
            return None

        axis.set_xlabel("prompt length (tokens)")
        axis.set_ylabel("time to first token (ms)")
        axis.set_xscale("log", base=2)
        # Direct labels run past the last point; leave them room instead of clipping.
        axis.margins(x=0.18)
        self._save(plt, figure, name)
        return name

    def quant_frontier(self, name: str = "fig_quant") -> Optional[str]:
        """Decode rate against model size across quantisations: the tradeoff curve."""
        plt = _require_matplotlib()
        _style(plt)
        usable = self.dataset.having("quant", "decode_tps", "weight_bytes")
        groups = [(key, subset) for key, subset in usable.group_by("device", "family")
                  if len(subset.unique("quant")) >= 2]
        if not groups:
            return None

        figure, axis = plt.subplots(figsize=(4.2, 3.2))
        for (device, family), subset in groups:
            best: Dict[str, Dict[str, Any]] = {}
            for row in subset.rows:
                current = best.get(row["quant"])
                if current is None or row["decode_tps"] > current["decode_tps"]:
                    best[row["quant"]] = row
            rows = sorted(best.values(), key=lambda r: r["weight_bytes"])
            sizes = [r["weight_bytes"] / 1e6 for r in rows]
            rates = [r["decode_tps"] for r in rows]
            color = self.color(device)
            axis.plot(sizes, rates, color=color, marker=self.marker(device),
                      markeredgecolor="white", markeredgewidth=0.6)
            for row, x, y in zip(rows, sizes, rates):
                axis.annotate(row["quant"], xy=(x, y), xytext=(0, 6),
                              textcoords="offset points", fontsize=7,
                              color=_MUTED, ha="center")
            axis.annotate(device, xy=(sizes[-1], rates[-1]), xytext=(6, -2),
                          textcoords="offset points", fontsize=8, color=color)

        axis.set_xlabel("model weights (MB)")
        axis.set_ylabel("decode throughput (tok/s)")
        axis.margins(x=0.15)
        self._save(plt, figure, name)
        return name

    def energy_per_token(self, name: str = "fig_energy") -> Optional[str]:
        """Energy per token by model, grouped by device. Bars, labelled directly."""
        plt = _require_matplotlib()
        _style(plt)
        usable = self.dataset.having("energy_per_token_mj")
        if not usable.rows:
            return None

        # One bar per (device, model, quant), best (lowest energy) thread config.
        best: Dict[Tuple, Dict[str, Any]] = {}
        for row in usable.rows:
            key = (row["device"], row["model"])
            current = best.get(key)
            if current is None or row["energy_per_token_mj"] < current["energy_per_token_mj"]:
                best[key] = row
        rows = sorted(best.values(), key=lambda r: (r["device"], r.get("weight_bytes") or 0))
        if not rows:
            return None

        figure, axis = plt.subplots(figsize=(4.6, 3.0))
        positions = range(len(rows))
        colors = [self.color(r["device"]) for r in rows]
        values = [r["energy_per_token_mj"] for r in rows]
        bars = axis.bar(positions, values, color=colors, width=0.62, zorder=3)
        for bar, value in zip(bars, values):
            axis.annotate(f"{value:.0f}", xy=(bar.get_x() + bar.get_width() / 2, value),
                          xytext=(0, 3), textcoords="offset points",
                          ha="center", fontsize=7.5, color=_INK)
        axis.set_xticks(list(positions))
        axis.set_xticklabels([r["model"] for r in rows], rotation=30, ha="right",
                             fontsize=7.5)
        devices = []
        for row in rows:
            if row["device"] not in devices:
                devices.append(row["device"])
        if len(devices) > 1:
            handles = [plt.Rectangle((0, 0), 1, 1, color=self.color(d)) for d in devices]
            axis.legend(handles, devices, loc="upper left")
        axis.set_ylabel("energy per token (mJ)")
        axis.grid(axis="x", visible=False)
        self._save(plt, figure, name)
        return name

    def vision_latency(self, name: str = "fig_vision") -> Optional[str]:
        """ONNX model latency: mean with a p95 whisker, grouped by device."""
        plt = _require_matplotlib()
        _style(plt)
        usable = self.dataset.having("latency_ms")
        if not usable.rows:
            return None

        best: Dict[Tuple, Dict[str, Any]] = {}
        for row in usable.rows:
            key = (row["device"], row["model"])
            current = best.get(key)
            if current is None or row["latency_ms"] < current["latency_ms"]:
                best[key] = row
        rows = sorted(best.values(), key=lambda r: (r["device"], r["latency_ms"]))
        if not rows:
            return None

        figure, axis = plt.subplots(figsize=(4.6, 3.0))
        positions = range(len(rows))
        values = [r["latency_ms"] for r in rows]
        p95s = [r.get("latency_p95") for r in rows]
        colors = [self.color(r["device"]) for r in rows]
        axis.bar(positions, values, color=colors, width=0.62, zorder=3)
        for index, (value, p95) in enumerate(zip(values, p95s)):
            if p95 and p95 > value:
                axis.plot([index, index], [value, p95], color=_INK, linewidth=1.0,
                          zorder=4)
                axis.plot([index], [p95], marker="_", color=_INK, markersize=8,
                          zorder=4)
        axis.set_xticks(list(positions))
        axis.set_xticklabels([f"{r['model']}" for r in rows], rotation=30, ha="right",
                             fontsize=7.5)
        devices = []
        for row in rows:
            if row["device"] not in devices:
                devices.append(row["device"])
        if len(devices) > 1:
            handles = [plt.Rectangle((0, 0), 1, 1, color=self.color(d)) for d in devices]
            axis.legend(handles, devices, loc="upper left")
        axis.set_ylabel("inference latency (ms), mean with p95 whisker")
        axis.grid(axis="x", visible=False)
        self._save(plt, figure, name)
        return name

    def cross_platform(self, name: str = "fig_crossplatform") -> Optional[str]:
        """Every device's roofline on one log-log plot: parallel lines, different heights.

        The figure that says the mechanism is platform-independent even though the
        machines differ by an order of magnitude.
        """
        devices = self.dataset.unique("device")
        if len(devices) < 2:
            return None
        return self.roofline(name=name)

    # ----------------------------------------------------------------------- all

    def generate_all(self) -> List[str]:
        produced = []
        for method in (self.roofline, self.thread_scaling, self.ttft,
                       self.quant_frontier, self.energy_per_token,
                       self.vision_latency):
            result = method()
            if result:
                produced.append(result)
        return produced
