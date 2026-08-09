"""A directory of run records, flattened into something you can filter and group.

Deliberately not pandas. The whole dataset is a few thousand rows at most, the operations
needed are filter, group and sort, and adding a heavy dependency to the analysis layer
would make the "clone it and reproduce a run" path meaningfully worse. numpy is used only
where there is real arithmetic to do.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..schema import RunRecord, load_records

# Flattened column names, chosen so that a table or a plot can name one string rather
# than walking three levels of nested record.
_COLUMNS = {
    "run_id": lambda r: r.run_id,
    "experiment": lambda r: r.experiment,
    "status": lambda r: r.status,
    "timestamp": lambda r: r.timestamp_utc,
    "duration_s": lambda r: r.duration_s,
    "device": lambda r: r.device.device_id,
    "cpu": lambda r: r.device.cpu,
    "board": lambda r: r.device.board_model,
    "arch": lambda r: r.device.machine,
    "cores": lambda r: r.device.physical_cores,
    "ram_bytes": lambda r: r.device.ram_bytes,
    "dram_peak_GBs": lambda r: r.device.dram_peak_GBs,
    "backend": lambda r: r.backend.name,
    "backend_version": lambda r: r.backend.version,
    "provider": lambda r: r.backend.execution_provider,
    "model": lambda r: r.workload.model,
    "quant": lambda r: r.workload.quantization,
    "family": lambda r: r.workload.architecture,
    "params_B": lambda r: r.workload.params,
    "weight_bytes": lambda r: r.workload.weight_bytes or r.workload.model_bytes,
    "prompt_tokens": lambda r: r.workload.prompt_tokens,
    "output_tokens": lambda r: r.workload.output_tokens,
    "context_tokens": lambda r: r.workload.context_tokens,
    "batch": lambda r: r.workload.batch_size,
    "input_shape": lambda r: r.workload.input_shape,
    "threads": lambda r: r.knobs.threads,
    "governor": lambda r: r.knobs.governor,
    "freq_khz": lambda r: r.knobs.freq_khz,
    "ttft_ms": lambda r: r.metrics.ttft_ms,
    "prefill_tps": lambda r: r.metrics.prefill_tps,
    "decode_tps": lambda r: r.metrics.decode_tps,
    "e2e_ms": lambda r: r.metrics.e2e_latency_ms,
    "load_ms": lambda r: r.metrics.load_ms,
    "latency_ms": lambda r: r.metrics.latency_ms_mean,
    "latency_p50": lambda r: r.metrics.latency_ms_p50,
    "latency_p95": lambda r: r.metrics.latency_ms_p95,
    "latency_p99": lambda r: r.metrics.latency_ms_p99,
    "ips": lambda r: r.metrics.throughput_ips,
    "decode_bw_GBs": lambda r: r.metrics.decode_bw_GBs,
    "bw_util_pct": lambda r: r.metrics.bw_utilization_pct,
    "stdev_pct": lambda r: r.metrics.stdev_pct,
    "power_w": lambda r: r.telemetry.power_w_mean,
    "power_core_w": lambda r: r.telemetry.power_w_core,
    "power_dram_w": lambda r: r.telemetry.power_w_dram,
    "energy_j": lambda r: r.telemetry.energy_j,
    "energy_per_token_mj": lambda r: r.telemetry.energy_per_token_mj,
    "temp_c": lambda r: r.telemetry.temp_c_max,
    "throttled": lambda r: r.telemetry.throttled,
    "cpu_util_pct": lambda r: r.telemetry.cpu_util_pct_mean,
    "mode": lambda r: "latency" if r.metrics.ttft_ms is not None else "throughput",
}


def flatten(record: RunRecord) -> Dict[str, Any]:
    row = {name: getter(record) for name, getter in _COLUMNS.items()}
    # Derived columns that are worth computing once rather than in every caller.
    if row["weight_bytes"]:
        row["weight_MB"] = row["weight_bytes"] / 1e6
    if row["decode_tps"] and row["power_w"]:
        row["energy_per_token_mj"] = row["energy_per_token_mj"] or (
            1000.0 * row["power_w"] / row["decode_tps"]
        )
    if row["latency_ms"] and row["power_w"]:
        row["energy_per_inference_mj"] = row["power_w"] * row["latency_ms"]
    return row


class Dataset:
    """A queryable view over run records."""

    def __init__(self, records: Sequence[RunRecord]):
        self.records = list(records)
        self.rows = [flatten(r) for r in self.records]

    # ------------------------------------------------------------------ factories

    @classmethod
    def from_directory(cls, path: str, include_failed: bool = False) -> "Dataset":
        records = load_records(path)
        if not include_failed:
            records = [r for r in records if r.ok]
        return cls(records)

    # ------------------------------------------------------------------- querying

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def filter(self, predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
               **equals: Any) -> "Dataset":
        """Filter by keyword equality, by predicate, or both."""
        kept_rows, kept_records = [], []
        for row, record in zip(self.rows, self.records):
            if any(row.get(k) != v for k, v in equals.items()):
                continue
            if predicate and not predicate(row):
                continue
            kept_rows.append(row)
            kept_records.append(record)
        result = Dataset.__new__(Dataset)
        result.records = kept_records
        result.rows = kept_rows
        return result

    def having(self, *columns: str) -> "Dataset":
        """Keep only rows where every named column has a value.

        The workhorse of this layer: most analyses are only meaningful over the subset of
        runs that actually measured the thing being analysed, and silently treating a
        missing measurement as zero is the failure mode this guards against.
        """
        return self.filter(lambda row: all(row.get(c) is not None for c in columns))

    def group_by(self, *columns: str) -> List[Tuple[Tuple, "Dataset"]]:
        buckets: Dict[Tuple, List[int]] = {}
        for index, row in enumerate(self.rows):
            key = tuple(row.get(c) for c in columns)
            buckets.setdefault(key, []).append(index)
        groups = []
        for key in sorted(buckets, key=_sort_key):
            indices = buckets[key]
            subset = Dataset.__new__(Dataset)
            subset.rows = [self.rows[i] for i in indices]
            subset.records = [self.records[i] for i in indices]
            groups.append((key, subset))
        return groups

    def sort(self, *columns: str, reverse: bool = False) -> "Dataset":
        order = sorted(range(len(self.rows)),
                       key=lambda i: _sort_key(tuple(self.rows[i].get(c) for c in columns)),
                       reverse=reverse)
        result = Dataset.__new__(Dataset)
        result.rows = [self.rows[i] for i in order]
        result.records = [self.records[i] for i in order]
        return result

    def values(self, column: str) -> List[Any]:
        return [row[column] for row in self.rows if row.get(column) is not None]

    def unique(self, column: str) -> List[Any]:
        seen: List[Any] = []
        for row in self.rows:
            value = row.get(column)
            if value is not None and value not in seen:
                seen.append(value)
        return sorted(seen, key=_sort_key)

    def best(self, column: str, maximise: bool = True) -> Optional[Dict[str, Any]]:
        candidates = [row for row in self.rows if row.get(column) is not None]
        if not candidates:
            return None
        return (max if maximise else min)(candidates, key=lambda row: row[column])

    # -------------------------------------------------------------------- warnings

    def integrity_warnings(self) -> List[str]:
        """Conditions that make numbers in this dataset less trustworthy.

        Surfaced rather than silently tolerated. A throttled run and a run competing with
        background load both produce numbers that look ordinary and are not comparable to
        the rest of the sweep.
        """
        notes: List[str] = []
        throttled = [r for r in self.rows if r.get("throttled")]
        if throttled:
            notes.append(f"{len(throttled)} run(s) throttled; their clocks were capped")

        noisy = [r for r in self.rows if (r.get("stdev_pct") or 0) > 10]
        if noisy:
            notes.append(
                f"{len(noisy)} run(s) had run-to-run spread above 10%, which usually means "
                "the machine was not idle"
            )

        over = [r for r in self.rows if (r.get("bw_util_pct") or 0) > 100]
        if over:
            notes.append(
                f"{len(over)} run(s) exceeded the declared memory bandwidth ceiling, so "
                "that ceiling is too low rather than the measurement being impossible"
            )

        failed = [r for r in self.rows if r.get("status") != "ok"]
        if failed:
            notes.append(f"{len(failed)} failed run(s) included in this view")
        return notes


# --------------------------------------------------------------------- statistics

def linear_fit_through_origin(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, float]:
    """Fit y = k*x and report R squared, for the decode roofline.

    Through the origin on purpose. The model is that decode reads the whole weight set
    once per token, so zero bytes means infinite tokens per second and a free intercept
    would be fitting a quantity with no physical meaning. R squared is computed against
    the mean of y, so it stays comparable to an ordinary regression.
    """
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys)
             if x is not None and y is not None and x > 0]
    if len(pairs) < 2:
        return {"slope": float("nan"), "r2": float("nan"), "n": len(pairs)}

    numerator = sum(x * y for x, y in pairs)
    denominator = sum(x * x for x, _ in pairs)
    slope = numerator / denominator if denominator else float("nan")

    mean_y = sum(y for _, y in pairs) / len(pairs)
    ss_residual = sum((y - slope * x) ** 2 for x, y in pairs)
    ss_total = sum((y - mean_y) ** 2 for _, y in pairs)
    r2 = 1.0 - ss_residual / ss_total if ss_total > 0 else float("nan")
    return {"slope": slope, "r2": r2, "n": len(pairs)}


def decode_roofline(dataset: "Dataset") -> Dict[str, Any]:
    """Fit tokens per second against inverse model size, the memory-wall relationship.

    Decode is bandwidth bound when tok/s = B / bytes for a constant B. Fitting tok/s
    against 1/bytes recovers B directly as the slope, in bytes per second.

    One point per model: the best decode rate across thread counts. The roofline is a
    statement about the ceiling, and folding sub-saturated thread points into the fit
    smears a thread effect into a size effect and wrecks R squared for the wrong reason.
    """
    usable = dataset.having("decode_tps", "weight_bytes")
    best_per_model: Dict[Any, Dict[str, Any]] = {}
    for row in usable.rows:
        key = (row.get("model"), row["weight_bytes"])
        current = best_per_model.get(key)
        if current is None or row["decode_tps"] > current["decode_tps"]:
            best_per_model[key] = row
    rows = list(best_per_model.values())
    xs = [1.0 / row["weight_bytes"] for row in rows]
    ys = [row["decode_tps"] for row in rows]
    fit = linear_fit_through_origin(xs, ys)
    fit["bw_eff_GBs"] = fit["slope"] / 1e9 if not math.isnan(fit["slope"]) else float("nan")

    peaks = usable.values("dram_peak_GBs")
    if peaks and not math.isnan(fit["bw_eff_GBs"]):
        peak = max(peaks)
        fit["dram_peak_GBs"] = peak
        fit["peak_fraction_pct"] = 100.0 * fit["bw_eff_GBs"] / peak
    return fit


def _sort_key(value: Any) -> Any:
    """Sort tuples that mix None, numbers and strings without raising."""
    if isinstance(value, tuple):
        return tuple(_sort_key(v) for v in value)
    if value is None:
        return (0, 0.0, "")
    if isinstance(value, bool):
        return (1, float(value), "")
    if isinstance(value, (int, float)):
        return (1, float(value), "")
    return (2, 0.0, str(value))
