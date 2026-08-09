"""ONNX Runtime backend, for the vision and classification half of the lab.

Generalised from the per-paper ``bench_infer.py`` scripts. The measurement itself happens
in :mod:`mlsyslab.bench_onnx`, launched by the agent as a subprocess, so ONNX Runtime is
never imported by the orchestrator or by the agent. That matters in both directions: the
host does not need the library to drive a device that has it, and a device without it
produces a clean failed record rather than an import error inside the agent.

The interpreter is per device. The Raspberry Pi's system python3 has ONNX Runtime 1.24;
this laptop's default python is 3.14 and has none, so its device config points at a 3.11
virtual environment. That is exactly the kind of difference the device abstraction exists
to absorb.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from ..bench_onnx import RESULT_BEGIN, RESULT_END
from ..schema import BackendInfo, RunRecord
from .base import Backend, BackendError, RunSpec
from .llamacpp import _base_record


class OnnxRuntimeBackend(Backend):
    name = "onnxruntime"

    # ------------------------------------------------------------------ discovery

    def discover(self, device) -> BackendInfo:
        """Ask the device's interpreter what it has, rather than assuming."""
        probe = (
            "import json,onnxruntime as ort;"
            "print(json.dumps({'version': ort.__version__,"
            "'providers': ort.get_available_providers()}))"
        )
        result = device.execute({
            "kind": "command",
            "argv": [device.python_executable, "-c", probe],
            "timeout_s": 180,
            "sample_power": False,
        })
        if result.get("status") != "ok":
            raise BackendError(
                f"onnxruntime is not usable on {device.device_id} via "
                f"{device.python_executable}: {result.get('error')}. "
                f"{(result.get('stderr') or '')[-300:]}"
            )
        try:
            payload = json.loads((result.get("stdout") or "").strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise BackendError(f"could not read onnxruntime version: {exc}") from exc

        return BackendInfo(
            name=self.name,
            version=payload.get("version"),
            binary=device.python_executable,
            execution_provider=(payload.get("providers") or ["CPUExecutionProvider"])[0],
            library_versions={"onnxruntime": payload.get("version", "unknown"),
                              "available_providers": ",".join(payload.get("providers") or [])},
        )

    def supports(self, spec: RunSpec) -> bool:
        return spec.mode in ("throughput", "latency")

    # ---------------------------------------------------------------- task building

    def build_task(self, spec: RunSpec, device) -> Dict[str, Any]:
        if not spec.model_path:
            raise BackendError(f"spec for model '{spec.model}' has no model_path")
        model_path = device.resolve(spec.model_path)
        if not device.exists(model_path):
            raise BackendError(f"model not found on {device.device_id}: {model_path}")

        root = device.package_root()
        argv = [device.python_executable, "-m", "mlsyslab.bench_onnx",
                "--model", model_path,
                "--reps", str(spec.extra.get("reps", spec.repetitions or 50)),
                "--warmup", str(spec.warmup if spec.warmup is not None else 10),
                "--batch", str(spec.batch_size or 1)]
        if spec.threads:
            argv += ["--threads", str(spec.threads)]
        for provider in (self.config.get("providers") or spec.extra.get("providers") or []):
            argv += ["--providers", provider]
        if spec.extra.get("optimization"):
            argv += ["--optimization", str(spec.extra["optimization"])]
        if spec.extra.get("dim_overrides"):
            argv += ["--dim-overrides", json.dumps(spec.extra["dim_overrides"])]
        if spec.extra.get("keep_samples"):
            argv += ["--keep-samples"]

        prepare = dict(spec.prepare)
        if spec.governor:
            prepare["governor"] = spec.governor
        if spec.freq_khz:
            prepare["freq_khz"] = spec.freq_khz

        return {
            "kind": "command",
            "argv": argv,
            "env": {"PYTHONPATH": root},
            "cwd": root,
            "timeout_s": spec.extra.get("timeout_s", 3600),
            "sample_hz": spec.sample_hz,
            "sample_power": spec.sample_power,
            # The warmup iterations run inside the subprocess before timing starts, so
            # the sampled window has to skip them too or power averages in the loader.
            "skip_leading_s": spec.extra.get("skip_leading_s", 0),
            "prepare": prepare,
        }

    # --------------------------------------------------------------------- parsing

    def parse(self, spec: RunSpec, result: Dict[str, Any], device) -> RunRecord:
        record = _base_record(spec, result, device, self.name)
        payload = _extract(result.get("stdout") or "")

        if payload is None:
            if record.status == "ok":
                record.status = "failed"
                record.error = "bench_onnx produced no parseable result"
            return record
        if payload.get("status") != "ok":
            record.status = "failed"
            record.error = payload.get("error") or "bench_onnx reported failure"
            return record

        record.backend.version = payload.get("onnxruntime_version")
        providers = payload.get("providers_used") or []
        record.backend.execution_provider = providers[0] if providers else None
        if payload.get("providers_unavailable"):
            record.warnings.append(
                "requested execution providers unavailable on this device: "
                + ", ".join(payload["providers_unavailable"])
            )

        metrics = record.metrics
        metrics.load_ms = payload.get("load_ms")
        metrics.latency_ms_mean = payload.get("latency_ms_mean")
        metrics.latency_ms_p50 = payload.get("latency_ms_p50")
        metrics.latency_ms_p95 = payload.get("latency_ms_p95")
        metrics.latency_ms_p99 = payload.get("latency_ms_p99")
        metrics.throughput_ips = payload.get("throughput_ips")
        if payload.get("latency_ms_mean"):
            stdev = payload.get("latency_ms_stdev") or 0.0
            metrics.stdev_pct = 100.0 * stdev / payload["latency_ms_mean"]
        metrics.extra.update({
            k: payload[k] for k in ("latency_ms_min", "latency_ms_max", "reps", "warmup")
            if payload.get(k) is not None
        })

        inputs = payload.get("inputs") or []
        if inputs and inputs[0].get("shape"):
            record.workload.input_shape = inputs[0]["shape"]
            if record.workload.batch_size is None:
                record.workload.batch_size = inputs[0]["shape"][0]
        record.workload.architecture = record.workload.architecture or _guess_family(spec.model)
        if payload.get("model"):
            record.workload.model_path = payload["model"]
        size = _model_size(device, payload.get("model"))
        if size:
            record.workload.model_bytes = size

        # Prompt and output token counts are meaningless for a vision model, and leaving
        # the RunSpec defaults in place would make them look measured.
        record.workload.prompt_tokens = None
        record.workload.output_tokens = None

        if payload.get("samples_ms"):
            record.raw["samples_ms"] = payload["samples_ms"]
        return record


# ------------------------------------------------------------------------ helpers

def _extract(stdout: str) -> Optional[Dict[str, Any]]:
    start = stdout.find(RESULT_BEGIN)
    end = stdout.find(RESULT_END)
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(stdout[start + len(RESULT_BEGIN):end].strip())
    except ValueError:
        return None


def _model_size(device, path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    # Only the local device can stat cheaply; a remote stat would be another SSH round
    # trip per run, and the size is a convenience rather than a measurement.
    try:
        if device.kind == "local" and os.path.exists(path):
            return os.path.getsize(path)
    except OSError:
        return None
    return None


def _guess_family(model_name: str) -> Optional[str]:
    lowered = (model_name or "").lower()
    for family in ("mobilenet_v3_small", "mobilenet_v3", "mobilenet", "resnet18",
                   "resnet50", "resnet", "efficientnet", "vit", "yolo"):
        if family in lowered:
            return family
    return None
