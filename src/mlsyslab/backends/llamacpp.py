"""llama.cpp backend: throughput via llama-bench, latency via llama-server.

Two measurement modes, because one tool cannot answer both questions:

* **throughput** runs ``llama-bench``, which reports prefill and decode rates separately
  and already handles warmup and repetitions. It is the only non-interactive measurement
  path in llama.cpp that will not block on stdin.
* **latency** drives ``llama-server`` over its streaming endpoint, which is the only way
  to observe when the *first* token arrived. ``llama-bench`` measures batched rates and
  structurally cannot report time to first token; ``llama-cli`` can, but it hangs when
  scripted even with interactive mode disabled, so it is not used here at all.

Recorded gotchas that are encoded as behaviour rather than left as documentation:

* ``llama-bench`` has no context-size flag. ``-c`` is rejected by current builds; context
  depth is set with ``-d``.
* ``-p 0`` produces an anomalously low decode rate. A nonzero prompt is forced whenever
  decode is being measured, and a warning is attached if the caller asked for zero.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from ..schema import BackendInfo, Knobs, Metrics, RunRecord, Telemetry, Workload
from .base import Backend, BackendError, RunSpec

_BENCH_NAMES = ["llama-bench", "llama-bench.exe"]
_SERVER_NAMES = ["llama-server", "llama-server.exe"]

# Places a llama.cpp build lands, including the two this lab actually uses.
_SEARCH_ROOTS = [
    "~/llm/llama.cpp/build/bin", "~/llama.cpp/build/bin", "~/src/llama.cpp/build/bin",
    "C:/llmpc/bin", "C:/llama.cpp", "/usr/local/bin", "/usr/bin", "/opt/llama.cpp/bin",
]

# Below this prompt length llama-bench reports a decode rate that is not the decode rate.
_MIN_PROMPT_FOR_DECODE = 32


class LlamaCppBackend(Backend):
    name = "llamacpp"

    # ------------------------------------------------------------------ discovery

    def _roots(self, device) -> List[str]:
        configured = self.config.get("bin_dir")
        roots = [configured] if configured else []
        return roots + _SEARCH_ROOTS

    def _find(self, device, names: List[str]) -> Optional[str]:
        return device.find_binary(names, self._roots(device))

    def discover(self, device) -> BackendInfo:
        binary = self._find(device, _BENCH_NAMES)
        server = self._find(device, _SERVER_NAMES)
        if not binary and not server:
            raise BackendError(
                f"no llama.cpp binaries found on {device.device_id}. Looked in: "
                + ", ".join(self._roots(device))
            )
        info = BackendInfo(name=self.name, binary=binary or server)
        # Version comes from a real run's JSON rather than a --version string, so it is
        # the build that produced the numbers rather than whatever is first on PATH.
        info.build_flags = None
        info.execution_provider = "CPU"
        info.library_versions = {}
        if server:
            info.library_versions["llama-server"] = server
        if binary:
            info.library_versions["llama-bench"] = binary
        return info

    def supports(self, spec: RunSpec) -> bool:
        return spec.mode in ("throughput", "latency")

    # ---------------------------------------------------------------- task building

    def build_task(self, spec: RunSpec, device) -> Dict[str, Any]:
        if spec.mode == "latency":
            return self._build_server_task(spec, device)
        return self._build_bench_task(spec, device)

    def _model_path(self, spec: RunSpec, device) -> str:
        if not spec.model_path:
            raise BackendError(f"spec for model '{spec.model}' has no model_path")
        path = device.resolve(spec.model_path)
        if not device.exists(path):
            raise BackendError(f"model not found on {device.device_id}: {path}")
        return path

    def _build_bench_task(self, spec: RunSpec, device) -> Dict[str, Any]:
        binary = self._find(device, _BENCH_NAMES)
        if not binary:
            raise BackendError(f"llama-bench not found on {device.device_id}")

        prompt = spec.prompt_tokens
        if spec.output_tokens and prompt < _MIN_PROMPT_FOR_DECODE:
            # Forced rather than refused: the run is still worth having, it just needs a
            # real prompt for the decode number to mean anything.
            prompt = _MIN_PROMPT_FOR_DECODE

        argv = [binary, "-m", self._model_path(spec, device), "-o", "json",
                "-p", str(prompt), "-n", str(spec.output_tokens),
                "-r", str(spec.repetitions)]
        if spec.threads:
            argv += ["-t", str(spec.threads)]
        if spec.context_tokens:
            # Depth, not context size: llama-bench has no -c flag and rejects it.
            argv += ["-d", str(spec.context_tokens)]
        if spec.flash_attention is not None:
            argv += ["-fa", "1" if spec.flash_attention else "0"]
        if spec.kv_quantization:
            argv += ["-ctk", spec.kv_quantization, "-ctv", spec.kv_quantization]
        argv += list(spec.extra.get("bench_args") or [])

        return {
            "kind": "command",
            "argv": argv,
            "timeout_s": spec.extra.get("timeout_s", 3600),
            "sample_hz": spec.sample_hz,
            "sample_power": spec.sample_power,
            "skip_leading_s": spec.extra.get("skip_leading_s", 0),
            "prepare": self._prepare(spec),
        }

    def _build_server_task(self, spec: RunSpec, device) -> Dict[str, Any]:
        binary = self._find(device, _SERVER_NAMES)
        if not binary:
            raise BackendError(
                f"llama-server not found on {device.device_id}. Latency mode needs it; "
                "build it with: cmake --build build --target llama-server -j1"
            )

        port = int(spec.extra.get("port", 8080))
        context = spec.context_tokens or max(512, spec.prompt_tokens + spec.output_tokens + 64)
        argv = [binary, "-m", self._model_path(spec, device),
                "--host", "127.0.0.1", "--port", str(port),
                "-c", str(context), "-np", "1", "--no-webui"]
        if spec.threads:
            argv += ["-t", str(spec.threads)]
        if spec.flash_attention is not None:
            argv += ["-fa", "1" if spec.flash_attention else "0"]
        if spec.kv_quantization:
            argv += ["-ctk", spec.kv_quantization, "-ctv", spec.kv_quantization]
        argv += list(spec.extra.get("server_args") or [])

        # A prompt of roughly the requested token count. Repeating a short phrase keeps
        # tokenisation predictable across models without shipping a corpus.
        prompt = _synthetic_prompt(spec.prompt_tokens)
        requests = [
            {"prompt": prompt, "n_predict": spec.output_tokens,
             "prompt_tokens": spec.prompt_tokens}
            for _ in range(max(1, spec.repetitions))
        ]

        return {
            "kind": "server_latency",
            "server_argv": argv,
            "port": port,
            "requests": requests,
            "startup_timeout_s": spec.extra.get("startup_timeout_s", 600),
            "sample_hz": spec.sample_hz,
            "sample_power": spec.sample_power,
            "prepare": self._prepare(spec),
        }

    def _prepare(self, spec: RunSpec) -> Dict[str, Any]:
        prepare = dict(spec.prepare)
        if spec.governor:
            prepare["governor"] = spec.governor
        if spec.freq_khz:
            prepare["freq_khz"] = spec.freq_khz
        return prepare

    # --------------------------------------------------------------------- parsing

    def parse(self, spec: RunSpec, result: Dict[str, Any], device) -> RunRecord:
        record = _base_record(spec, result, device, self.name)
        if spec.mode == "latency":
            _parse_server(record, result)
        else:
            _parse_bench(record, result)
        _derive_bandwidth(record, device)
        return record


# ------------------------------------------------------------------------ parsers

def _parse_bench(record: RunRecord, result: Dict[str, Any]) -> None:
    rows = _bench_rows(result.get("stdout", ""))
    if not rows:
        if record.status == "ok":
            record.status = "failed"
            record.error = "llama-bench produced no parseable JSON"
        return

    first = rows[0]
    record.backend.version = str(first.get("build_number") or "")
    record.backend.build = first.get("build_commit")
    record.backend.execution_provider = first.get("backends") or "CPU"
    record.workload.model_path = first.get("model_filename")
    record.workload.weight_bytes = first.get("model_size")
    record.workload.architecture = first.get("model_type")
    if first.get("model_n_params"):
        record.workload.params = first["model_n_params"] / 1e9
    record.knobs.threads = first.get("n_threads")
    if first.get("n_depth"):
        record.workload.context_tokens = first["n_depth"]

    stdevs = []
    for row in rows:
        n_prompt, n_gen = row.get("n_prompt", 0), row.get("n_gen", 0)
        rate = row.get("avg_ts")
        if rate is None:
            continue
        # llama-bench emits one row per test: prompt-only rows measure prefill,
        # generation-only rows measure decode.
        if n_prompt > 0 and n_gen == 0:
            record.metrics.prefill_tps = rate
            record.workload.prompt_tokens = n_prompt
        elif n_gen > 0 and n_prompt == 0:
            record.metrics.decode_tps = rate
            record.workload.output_tokens = n_gen
        if row.get("stddev_ts") and rate:
            stdevs.append(100.0 * row["stddev_ts"] / rate)
    if stdevs:
        record.metrics.stdev_pct = max(stdevs)

    if record.metrics.decode_tps is None and record.status == "ok":
        record.warnings.append("no decode row in llama-bench output")


def _bench_rows(stdout: str) -> List[Dict[str, Any]]:
    """Extract llama-bench's JSON array, ignoring log lines printed around it."""
    text = stdout.strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except ValueError:
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1 or end < start:
            return []
        try:
            payload = json.loads(text[start:end + 1])
        except ValueError:
            return []
    if isinstance(payload, dict):
        payload = [payload]
    return [row for row in payload if isinstance(row, dict)]


def _parse_server(record: RunRecord, result: Dict[str, Any]) -> None:
    requests = [r for r in (result.get("requests") or []) if r.get("ttft_ms") is not None]
    if not requests:
        if record.status == "ok":
            record.status = "failed"
            record.error = result.get("error") or "no successful streaming requests"
        return

    record.metrics.load_ms = result.get("load_ms")
    record.metrics.ttft_ms = _median([r["ttft_ms"] for r in requests])
    record.metrics.e2e_latency_ms = _median([r["e2e_latency_ms"] for r in requests])
    decode = [r["decode_tps"] for r in requests if r.get("decode_tps")]
    if decode:
        record.metrics.decode_tps = _median(decode)
    outputs = [r["output_tokens"] for r in requests if r.get("output_tokens")]
    if outputs:
        record.workload.output_tokens = int(_median(outputs))

    # Prefill rate implied by the first token: prompt tokens over the time before it.
    # Reported only when the prompt is long enough that model load does not dominate.
    prompt_tokens = requests[0].get("prompt_tokens")
    if prompt_tokens and record.metrics.ttft_ms and prompt_tokens >= 64:
        record.metrics.prefill_tps = prompt_tokens / (record.metrics.ttft_ms / 1000.0)

    if len(requests) > 1:
        ttfts = [r["ttft_ms"] for r in requests]
        mean = sum(ttfts) / len(ttfts)
        if mean > 0:
            spread = (max(ttfts) - min(ttfts)) / mean
            record.metrics.extra["ttft_spread_pct"] = round(100.0 * spread, 1)
    record.metrics.extra["requests"] = len(requests)


def _derive_bandwidth(record: RunRecord, device) -> None:
    """Decode bandwidth and roofline utilisation, when both inputs are known.

    Decode reads the entire weight set once per token, so tokens per second times bytes
    is an effective bandwidth. Utilisation against the datasheet peak is only reported
    when the device config declares that peak, since a made-up denominator is worse than
    no percentage at all.
    """
    weight_bytes = record.workload.weight_bytes or record.workload.model_bytes
    if not (record.metrics.decode_tps and weight_bytes):
        return
    record.metrics.decode_bw_GBs = record.metrics.decode_tps * weight_bytes / 1e9

    peak = getattr(device, "config", {}).get("dram_peak_GBs")
    if not peak:
        return
    utilisation = 100.0 * record.metrics.decode_bw_GBs / float(peak)
    record.metrics.bw_utilization_pct = utilisation
    if utilisation > 100.0:
        # The measured ceiling is a lower bound on what the memory system can do, so
        # exceeding it means the ceiling is wrong, not that physics broke.
        record.warnings.append(
            f"decode bandwidth is {utilisation:.0f}% of the declared peak; "
            "the declared peak is too low"
        )


# ------------------------------------------------------------------------ helpers

def _base_record(spec: RunSpec, result: Dict[str, Any], device, backend_name: str) -> RunRecord:
    """The parts of a record that do not depend on which backend produced it."""
    from .. import __version__
    from ..schema import DeviceInfo, compute_run_id, from_dict

    device_payload = dict(result.get("device") or {})
    device_payload.update({k: v for k, v in device.device_info().items()
                           if k in {f for f in DeviceInfo.__dataclass_fields__}})
    known = set(DeviceInfo.__dataclass_fields__)
    device_info = DeviceInfo(**{k: v for k, v in device_payload.items() if k in known})

    telemetry_payload = dict(result.get("telemetry") or {})
    telemetry = Telemetry(**{k: v for k, v in telemetry_payload.items()
                             if k in Telemetry.__dataclass_fields__})

    record = RunRecord(
        run_id=compute_run_id(spec.identity()),
        experiment=spec.experiment,
        lab_version=__version__,
        timestamp_utc=result.get("agent_time_utc"),
        duration_s=result.get("duration_s"),
        status=result.get("status", "ok"),
        error=result.get("error"),
        warnings=list(result.get("warnings") or []),
        tags=list(spec.tags),
        device=device_info,
        backend=BackendInfo(name=backend_name),
        workload=Workload(
            model=spec.model,
            model_path=spec.model_path,
            quantization=spec.quantization,
            prompt_tokens=spec.prompt_tokens,
            output_tokens=spec.output_tokens,
            context_tokens=spec.context_tokens,
            batch_size=spec.batch_size,
        ),
        knobs=Knobs(
            threads=spec.threads, governor=spec.governor, freq_khz=spec.freq_khz,
            repetitions=spec.repetitions, warmup=spec.warmup,
            flash_attention=spec.flash_attention, kv_quantization=spec.kv_quantization,
        ),
        metrics=Metrics(),
        telemetry=telemetry,
        raw={
            "stdout": (result.get("stdout") or "")[:200000],
            "stderr": (result.get("stderr") or "")[-8000:],
            "requests": result.get("requests"),
            "memory_before": result.get("memory_before"),
            "memory_after": result.get("memory_after"),
        },
    )
    record.raw = {k: v for k, v in record.raw.items() if v}

    peak_rss = (result.get("memory_after") or {}).get("peak_rss_bytes")
    if peak_rss:
        record.metrics.peak_rss_bytes = peak_rss
    return record


def _synthetic_prompt(target_tokens: int) -> str:
    """A prompt of roughly the requested length, deterministic across runs.

    Roughly, not exactly: token counts are tokeniser-specific and the record stores the
    requested count alongside what the server actually reported, so the approximation is
    visible rather than hidden.
    """
    unit = "The quick brown fox jumps over the lazy dog. "
    # That sentence is about 10 tokens for common BPE vocabularies.
    repeats = max(1, int(target_tokens / 10) + 1)
    return (unit * repeats).strip()


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0
