"""Publish what a sweep is doing, without letting the dashboard become the data.

The run records on disk are the result. This module is a window onto a sweep while it is
happening: which machines are up, what has finished, what the last point on each device
measured. Three rules keep it a window rather than a second, worse copy of the science.

**The exporter can die and nothing is lost.** It observes; it is never in the write path.
Every number here was already written to ``runs/.../*.json`` before it reached this
registry, so a scrape that never happens costs a graph, not a measurement.

**No new dependency.** The text exposition format is a hundred lines of specification and
about fifty of code, and the alternative is putting ``prometheus_client`` in the
dependency list of a package whose measurement core is deliberately standard library
only. A convenience for watching does not get to change what has to be installed on a
2 GB board.

**Labels are bounded by the config.** Device, backend, model, quantization and mode all
come from a config file with a known number of values. Run ids and timestamps do not
appear as labels, because a benchmark that runs for a week would otherwise leave a
time series database with a million dead series in it.

What this is not: a live power trace. Telemetry is sampled on the device, inside the
agent, and comes back when the run does. A point appears here when it finishes, which is
the right granularity for watching a sweep and the wrong one for watching a waveform.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

_NAME_PREFIX = "mlsyslab_"

# name -> (type, help). Declared up front so that every series that does appear carries
# its own explanation. A metric with no samples yet is not emitted at all: an endpoint
# full of empty declarations reads as "measured and zero", which is the one thing this
# codebase refuses to say about a number nobody has taken.
_METRICS: Dict[str, Tuple[str, str]] = {
    "build_info": ("gauge", "Version of ml-systems-lab producing these metrics"),
    "sweep_runs_total": ("counter", "Runs that have finished, by device and status"),
    "sweep_runs_pending": ("gauge", "Runs queued for a device and not yet started"),
    "sweep_runs_inflight": ("gauge", "Runs executing on a device right now"),
    "sweep_runs_skipped_total": ("counter", "Runs skipped because a result already existed"),
    "sweep_elapsed_seconds": ("gauge", "Wall clock since the sweep started"),
    "device_up": ("gauge", "1 when a device is in service, 0 when its breaker is open"),
    "device_transport_failures_total": ("counter", "Transport failures seen on a device"),
    "device_retries_total": ("counter", "Retries issued for a device"),
    "device_clock_offset_seconds": ("gauge",
                                    "Device clock minus host clock, from the pre-sweep check"),
    "device_cpu_quota_cores": ("gauge",
                               "Effective cgroup CPU quota in cores, where one applies"),
    "decode_tokens_per_second": ("gauge", "Decode throughput of the last completed run"),
    "prefill_tokens_per_second": ("gauge", "Prefill throughput of the last completed run"),
    "ttft_milliseconds": ("gauge", "Time to first token of the last completed run"),
    "latency_milliseconds": ("gauge", "Mean single-shot latency of the last completed run"),
    "throughput_inferences_per_second": ("gauge", "Inference throughput of the last run"),
    "power_watts": ("gauge", "Mean package power during the last completed run"),
    "temperature_celsius": ("gauge", "Peak temperature during the last completed run"),
    "energy_per_token_millijoules": ("gauge", "Energy per generated token, last run"),
    "peak_rss_bytes": ("gauge", "Peak resident set size of the last completed run"),
    "bandwidth_utilization_ratio": ("gauge",
                                    "Achieved DRAM bandwidth over the device's peak, last run"),
    "run_throttled": ("gauge", "1 if the last completed run reported thermal throttling"),
}

# Which record field feeds which metric. Kept as data so adding a metric is one line and
# so nothing here has to know the shape of a RunRecord twice.
_FROM_METRICS = [
    ("decode_tokens_per_second", "decode_tps", 1.0),
    ("prefill_tokens_per_second", "prefill_tps", 1.0),
    ("ttft_milliseconds", "ttft_ms", 1.0),
    ("latency_milliseconds", "latency_ms_mean", 1.0),
    ("throughput_inferences_per_second", "throughput_ips", 1.0),
    ("peak_rss_bytes", "peak_rss_bytes", 1.0),
    ("bandwidth_utilization_ratio", "bw_utilization_pct", 0.01),
]
_FROM_TELEMETRY = [
    ("power_watts", "power_w_mean", 1.0),
    ("temperature_celsius", "temp_c_max", 1.0),
    ("energy_per_token_millijoules", "energy_per_token_mj", 1.0),
]


def _escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_labels(labels: Iterable[Tuple[str, str]]) -> str:
    pairs = [f'{k}="{_escape(v)}"' for k, v in labels if v not in (None, "")]
    return "{" + ",".join(pairs) + "}" if pairs else ""


class Registry:
    """Everything a scrape would report, updated as a sweep runs.

    Deliberately a plain dict of ``(metric, labels) -> value`` rather than anything
    clever. The whole state is small, a scrape renders all of it, and being able to print
    it in a debugger has been worth more than any efficiency this gives up.
    """

    def __init__(self, version: Optional[str] = None, experiment: str = "unnamed"):
        if version is None:
            from . import __version__ as version
        self.version = version
        self.experiment = experiment
        self._values: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
        self._lock = threading.Lock()
        self._started = time.time()
        self.set("build_info", 1.0, version=self.version)

    # ------------------------------------------------------------------- primitives

    def set(self, name: str, value: Optional[float], **labels: Any) -> None:
        if value is None:
            return
        key = (name, self._key(labels))
        with self._lock:
            self._values[key] = float(value)

    def add(self, name: str, delta: float = 1.0, **labels: Any) -> None:
        key = (name, self._key(labels))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + float(delta)

    def get(self, name: str, **labels: Any) -> Optional[float]:
        with self._lock:
            return self._values.get((name, self._key(labels)))

    def _key(self, labels: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
        merged = dict(labels)
        merged.setdefault("experiment", self.experiment)
        return tuple(sorted((k, str(v)) for k, v in merged.items() if v not in (None, "")))

    # ---------------------------------------------------------------- record intake

    def observe_record(self, record) -> None:
        """Publish one finished run. Never raises: watching must not break running."""
        try:
            labels = _labels_for(record)
            self.add("sweep_runs_total", 1.0, device=labels["device"],
                     status=record.status)
            if not record.ok:
                return
            for metric, field, scale in _FROM_METRICS:
                value = getattr(record.metrics, field, None)
                if value is not None:
                    self.set(metric, value * scale, **labels)
            for metric, field, scale in _FROM_TELEMETRY:
                value = getattr(record.telemetry, field, None)
                if value is not None:
                    self.set(metric, value * scale, **labels)
            if record.telemetry.throttled is not None:
                self.set("run_throttled", 1.0 if record.telemetry.throttled else 0.0,
                         **labels)
            quota = (record.device.placement or {}).get("cpu_quota_cores")
            if quota is not None:
                self.set("device_cpu_quota_cores", quota, device=labels["device"])
        except Exception:                       # pragma: no cover - defensive by design
            pass

    def observer(self) -> Callable[[str, Dict[str, Any]], None]:
        """An ``on_event`` callback that keeps this registry current during a sweep."""

        def observe(kind: str, payload: Dict[str, Any]) -> None:
            try:
                spec = payload.get("spec")
                device = getattr(spec, "device_id", None) or payload.get("device")
                self.set("sweep_elapsed_seconds", time.time() - self._started)
                if kind == "start" and device:
                    self.add("sweep_runs_inflight", 1.0, device=device)
                elif kind == "finish":
                    if device:
                        self.add("sweep_runs_inflight", -1.0, device=device)
                    self.observe_record(payload["record"])
                elif kind == "skipped" and device:
                    self.add("sweep_runs_skipped_total", 1.0, device=device)
                elif kind == "retry" and device:
                    self.add("device_retries_total", 1.0, device=device)
                elif kind == "device_down" and device:
                    self.set("device_up", 0.0, device=device)
                    self.add("device_transport_failures_total",
                             float(payload.get("consecutive_failures") or 1),
                             device=device)
                elif kind == "device_up" and device:
                    self.set("device_up", 1.0, device=device)
                elif kind == "clock" and device:
                    self.set("device_up", 1.0 if payload.get("status") == "ok" else 0.0,
                             device=device)
                    if payload.get("offset_s") is not None:
                        self.set("device_clock_offset_seconds", payload["offset_s"],
                                 device=device)
                elif kind == "abandoned" and payload.get("device"):
                    self.set("sweep_runs_pending", 0.0, device=payload["device"])
            except Exception:                   # pragma: no cover - defensive by design
                pass

        return observe

    def observe_devices(self, devices: Dict[str, Any]) -> None:
        """Record what constrains each device, so a graph can be read in context.

        A throughput chart with no CPU quota on it invites the reader to compare two
        numbers that were never comparable.
        """
        for device_id, device in devices.items():
            limits = getattr(device, "cgroup_limits", None)
            if not callable(limits):
                continue
            try:
                cores = limits().get("cpu_quota_cores")
            except Exception:
                continue
            if cores is not None:
                self.set("device_cpu_quota_cores", cores, device=device_id)

    def observe_queue(self, pending: Dict[str, int]) -> None:
        for device_id, count in pending.items():
            self.set("sweep_runs_pending", float(count), device=device_id)

    # ------------------------------------------------------------------- exposition

    def render(self) -> str:
        with self._lock:
            snapshot = dict(self._values)

        by_metric: Dict[str, List[Tuple[Tuple[Tuple[str, str], ...], float]]] = {}
        for (name, labels), value in snapshot.items():
            by_metric.setdefault(name, []).append((labels, value))

        lines: List[str] = []
        for name in sorted(by_metric):
            kind, help_text = _METRICS.get(name, ("gauge", "Undeclared metric"))
            full = _NAME_PREFIX + name
            lines.append(f"# HELP {full} {help_text}")
            lines.append(f"# TYPE {full} {kind}")
            for labels, value in sorted(by_metric[name]):
                lines.append(f"{full}{_render_labels(labels)} {_format(value)}")
        return "\n".join(lines) + "\n"


def _labels_for(record) -> Dict[str, Any]:
    """The identity of a series.

    Every one of these is bounded by the config, and every axis a sweep varies is here.
    Leaving one out would not reduce cardinality so much as silently collapse points that
    were never the same measurement onto one line of a graph, which is the specific way a
    dashboard starts lying. Run ids and timestamps are deliberately absent: those are
    unbounded, and they are what the records are for.

    ``mode`` is derived the same way the analysis layer derives it, rather than stored
    twice and allowed to disagree with itself.
    """
    workload = record.workload
    tokens = ""
    if workload.prompt_tokens is not None and workload.output_tokens is not None:
        tokens = f"p{workload.prompt_tokens}n{workload.output_tokens}"
    elif workload.batch_size is not None:
        tokens = f"b{workload.batch_size}"
    return {
        "device": record.device.device_id,
        "backend": record.backend.name,
        "model": workload.model,
        "quant": workload.quantization,
        "mode": "latency" if record.metrics.ttft_ms is not None else "throughput",
        "threads": record.knobs.threads,
        "tokens": tokens,
    }


def _format(value: float) -> str:
    if value != value:                                   # NaN
        return "NaN"
    if value in (float("inf"), float("-inf")):
        return "+Inf" if value > 0 else "-Inf"
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return repr(float(value))


def from_directory(path: str, experiment: Optional[str] = None) -> Registry:
    """Build a registry from run records already on disk.

    For the textfile collector, or for a one-shot scrape of a finished campaign. The
    records remain the source of truth; this is a projection of them.
    """
    from .schema import load_records

    records = load_records(path)
    registry = Registry(experiment=experiment or (records[0].experiment if records
                                                  else "unnamed"))
    for record in records:
        registry.observe_record(record)
    return registry


# --------------------------------------------------------------------- http serving

class Exporter:
    """A /metrics endpoint on a background thread, for the duration of a sweep."""

    def __init__(self, registry: Registry, port: int = 9109, host: str = "127.0.0.1"):
        self.registry = registry
        self.host = host
        self.port = port
        self._server = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "Exporter":
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        registry = self.registry

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):                    # noqa: N802 - http.server's spelling
                if self.path.split("?")[0] not in ("/metrics", "/"):
                    self.send_error(404, "only /metrics is served here")
                    return
                body = registry.render().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type",
                                 "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                """Silence. A scrape every fifteen seconds is not news."""

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="mlsyslab-exporter", daemon=True)
        self._thread.start()
        return self

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/metrics"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "Exporter":
        return self.start() if self._server is None else self

    def __exit__(self, *_exc) -> None:
        self.stop()


def chain(*observers: Optional[Callable[[str, Dict[str, Any]], None]]):
    """Fan one event stream out to several listeners, e.g. the console and a registry."""
    live = [o for o in observers if o]

    def fan_out(kind: str, payload: Dict[str, Any]) -> None:
        for observer in live:
            observer(kind, payload)

    return fan_out
