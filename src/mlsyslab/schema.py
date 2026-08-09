"""The single record every measurement produces, whatever device or backend made it.

Design rules, learned the hard way from five one-off benchmark campaigns:

1. **One flat record per run.** Not a nested per-experiment blob. Analysis then means
   filtering a list, and a run from a Pi in 2026 stays comparable to a run from a laptop
   two years later.
2. **Every metric is optional and defaults to None.** A device that cannot read power
   reports None, not zero and not a crash. Zero is a measurement; None is an absence, and
   conflating them silently poisons every average downstream.
3. **Provenance travels with the number.** Backend version, build flags, kernel, compiler
   and library versions live in the record itself. A results directory has to be
   interpretable without the config that produced it.
4. **The record knows whether to trust itself.** ``status`` and ``warnings`` are part of
   the schema because a run that thermally throttled or fought background load is not a
   failed run, it is a run whose numbers mean something different.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"


def _clean(obj: Any) -> Any:
    """Recursively drop None values so records stay readable, keeping empty containers."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _clean(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


@dataclass
class DeviceInfo:
    """What the hardware is. Collected automatically, never typed in by hand."""

    device_id: str = "unknown"          # stable label, e.g. "pi5-2gb" or "laptop-i7-12700h"
    kind: str = "local"                 # local | ssh
    os: Optional[str] = None
    os_release: Optional[str] = None
    distro: Optional[str] = None
    kernel: Optional[str] = None
    machine: Optional[str] = None       # x86_64, aarch64
    cpu: Optional[str] = None
    cpu_flags: Optional[List[str]] = None   # only the ones that change kernel selection
    logical_cores: Optional[int] = None
    physical_cores: Optional[int] = None
    ram_bytes: Optional[int] = None
    swap_bytes: Optional[int] = None
    dram_peak_GBs: Optional[float] = None   # datasheet peak, for roofline utilisation
    dram_measured_GBs: Optional[float] = None  # best read-only kernel we could achieve
    board_model: Optional[str] = None       # Raspberry Pi 5 Model B Rev 1.0
    gpu: Optional[str] = None
    accelerator: Optional[str] = None
    python: Optional[str] = None
    compiler: Optional[str] = None
    libc: Optional[str] = None
    capabilities: Dict[str, bool] = field(default_factory=dict)


@dataclass
class BackendInfo:
    """What ran the model, precisely enough to explain a regression a year later."""

    name: str = "unknown"               # llamacpp | onnxruntime
    version: Optional[str] = None
    build: Optional[str] = None         # llama.cpp build number / commit
    binary: Optional[str] = None        # resolved path actually invoked
    execution_provider: Optional[str] = None   # CPUExecutionProvider, CUDA, ...
    build_flags: Optional[List[str]] = None
    library_versions: Dict[str, str] = field(default_factory=dict)


@dataclass
class Workload:
    """What was run. Sizes are in bytes because tokens per second divides by bytes."""

    model: str = "unknown"
    model_path: Optional[str] = None
    model_bytes: Optional[int] = None       # on-disk size
    weight_bytes: Optional[int] = None      # backend-reported size, preferred for roofline
    params: Optional[float] = None          # billions
    quantization: Optional[str] = None      # Q4_K_M, int8, fp32
    architecture: Optional[str] = None
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    context_tokens: Optional[int] = None    # KV depth the run was measured at
    batch_size: Optional[int] = None
    input_shape: Optional[List[int]] = None  # vision models
    kv_bytes_per_token: Optional[int] = None


@dataclass
class Knobs:
    """Everything the operator chose. These are the axes an experiment sweeps."""

    threads: Optional[int] = None
    governor: Optional[str] = None
    freq_khz: Optional[int] = None
    pinned: Optional[bool] = None
    repetitions: Optional[int] = None
    warmup: Optional[int] = None
    flash_attention: Optional[bool] = None
    kv_quantization: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metrics:
    """The numbers. All optional: a backend reports what it can measure and nothing more.

    Throughput and latency are kept side by side deliberately. llama-bench reports
    aggregate prefill and decode rates but cannot tell you when the first token arrived;
    a streaming server request can. A record may carry either, or both.
    """

    ttft_ms: Optional[float] = None              # time to first token, single request
    prefill_tps: Optional[float] = None          # prompt tokens per second
    decode_tps: Optional[float] = None           # generated tokens per second
    e2e_latency_ms: Optional[float] = None       # full request, submit to last token
    load_ms: Optional[float] = None              # model load / session init
    latency_ms_mean: Optional[float] = None      # single-shot inference (vision models)
    latency_ms_p50: Optional[float] = None
    latency_ms_p95: Optional[float] = None
    latency_ms_p99: Optional[float] = None
    throughput_ips: Optional[float] = None       # inferences per second
    peak_rss_bytes: Optional[int] = None
    model_resident_bytes: Optional[int] = None
    decode_bw_GBs: Optional[float] = None        # decode_tps * weight_bytes
    bw_utilization_pct: Optional[float] = None
    stdev_pct: Optional[float] = None            # run-to-run spread, stability signal
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Telemetry:
    """Physical state during the measured window.

    Temperature lives here rather than in a side CSV because a throttled run and a cool
    run are different measurements, and joining them after the fact by timestamp is how
    you end up quietly comparing the two.
    """

    power_w_mean: Optional[float] = None
    power_w_core: Optional[float] = None
    power_w_dram: Optional[float] = None
    idle_power_w: Optional[float] = None
    energy_j: Optional[float] = None
    energy_per_token_mj: Optional[float] = None
    temp_c_start: Optional[float] = None
    temp_c_mean: Optional[float] = None
    temp_c_max: Optional[float] = None
    throttled: Optional[bool] = None
    throttle_flags: Optional[str] = None
    cpu_util_pct_mean: Optional[float] = None
    cpu_util_pct_per_core: Optional[List[float]] = None
    freq_khz_mean: Optional[float] = None
    freq_khz_min: Optional[int] = None
    freq_khz_max: Optional[int] = None
    sample_hz: Optional[float] = None
    sample_count: Optional[int] = None


@dataclass
class RunRecord:
    """One measurement, fully self-describing."""

    run_id: str = ""
    experiment: str = "unnamed"
    schema_version: str = SCHEMA_VERSION
    lab_version: Optional[str] = None
    timestamp_utc: Optional[str] = None
    duration_s: Optional[float] = None
    status: str = "ok"                       # ok | failed | skipped
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    device: DeviceInfo = field(default_factory=DeviceInfo)
    backend: BackendInfo = field(default_factory=BackendInfo)
    workload: Workload = field(default_factory=Workload)
    knobs: Knobs = field(default_factory=Knobs)
    metrics: Metrics = field(default_factory=Metrics)
    telemetry: Telemetry = field(default_factory=Telemetry)

    # Whatever the backend actually printed, kept so a parsing bug found later can be
    # fixed by reparsing instead of by re-running an eight-hour campaign.
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _clean(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False, default=str)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def compute_run_id(*parts: Any) -> str:
    """A short, deterministic id for a run.

    Deterministic on purpose: re-running the same spec produces the same id, so a resumed
    campaign can skip work it already has instead of duplicating it. Timestamps are
    excluded for exactly that reason.
    """
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def from_dict(data: Dict[str, Any]) -> RunRecord:
    """Rebuild a RunRecord from JSON, tolerating fields added by newer versions.

    Unknown keys are dropped rather than raising: an analysis script from today must
    still be able to read a record written by a later version of the schema.
    """
    def build(cls, payload):
        if not isinstance(payload, dict):
            return cls()
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in known})

    known_top = {f.name for f in dataclasses.fields(RunRecord)}
    nested = {
        "device": DeviceInfo, "backend": BackendInfo, "workload": Workload,
        "knobs": Knobs, "metrics": Metrics, "telemetry": Telemetry,
    }
    kwargs = {k: v for k, v in data.items() if k in known_top and k not in nested}
    for key, cls in nested.items():
        kwargs[key] = build(cls, data.get(key, {}))
    return RunRecord(**kwargs)


def load_records(path) -> List[RunRecord]:
    """Read every RunRecord under a directory, or a single .json / .jsonl file."""
    import os

    files: List[str] = []
    if os.path.isdir(path):
        for root, _dirs, names in os.walk(path):
            for name in sorted(names):
                if name.endswith((".json", ".jsonl")):
                    files.append(os.path.join(root, name))
    else:
        files = [str(path)]

    records: List[RunRecord] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            continue
        if fp.endswith(".jsonl"):
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                # Only actual records. The runner's index.jsonl and other sidecar files
                # share the extension but are not measurements.
                if isinstance(payload, dict) and "schema_version" in payload:
                    try:
                        records.append(from_dict(payload))
                    except TypeError:
                        continue
            continue
        try:
            payload = json.loads(text)
        except ValueError:
            continue
        # A file may hold one record or a list of them.
        for item in payload if isinstance(payload, list) else [payload]:
            if isinstance(item, dict) and "schema_version" in item:
                records.append(from_dict(item))
    return records
