"""Execute a config, one run at a time, writing each result before starting the next.

Three properties matter more than speed here, because a full matrix is an overnight job:

* **Every run is written to disk immediately.** A campaign that dies at hour six still has
  five hours of results, and they are readable without the runner.
* **Runs are resumable by id.** Run ids are deterministic over the spec, so restarting
  skips completed points instead of redoing them.
* **A failure is a record, not an exception.** A model that will not fit in RAM produces a
  record with ``status: failed`` and the error in it. That is data about the machine, and
  it belongs in the results directory rather than in a terminal that has scrolled away.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

from . import backends as backend_registry
from . import devices as device_registry
from .backends.base import RunSpec
from .config import Config
from .schema import RunRecord, compute_run_id


class Runner:
    def __init__(
        self,
        config: Config,
        output_dir: Optional[str] = None,
        resume: bool = True,
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.config = config
        self.output_dir = os.path.abspath(output_dir or config.output_dir)
        self.resume = resume
        self.on_event = on_event or (lambda kind, payload: None)
        self._devices: Dict[str, Any] = {}
        self._backends: Dict[str, Any] = {}

    # --------------------------------------------------------------------- wiring

    def device(self, device_id: str):
        if device_id not in self._devices:
            self._devices[device_id] = device_registry.from_config(
                device_id, self.config.devices.get(device_id, {})
            )
        return self._devices[device_id]

    def backend(self, name: str, device_id: str):
        key = f"{name}@{device_id}"
        if key not in self._backends:
            device_config = self.config.devices.get(device_id, {})
            self._backends[key] = backend_registry.get(name, device_config.get(name) or {})
        return self._backends[key]

    # ---------------------------------------------------------------------- paths

    def record_path(self, spec: RunSpec) -> str:
        run_id = compute_run_id(spec.identity())
        return os.path.join(self.output_dir, spec.device_id, f"{run_id}.json")

    def already_done(self, spec: RunSpec) -> bool:
        """A completed run is one that exists on disk and did not fail.

        Failed runs are retried on resume: a failure is usually environmental (a busy
        machine, a server that would not start), and re-running is cheap next to
        silently carrying a hole in the matrix.
        """
        path = self.record_path(spec)
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh).get("status") == "ok"
        except (OSError, ValueError):
            return False

    # ------------------------------------------------------------------ execution

    def run(self, specs: Optional[List[RunSpec]] = None) -> List[RunRecord]:
        specs = specs if specs is not None else self.config.specs
        os.makedirs(self.output_dir, exist_ok=True)
        self._write_manifest(specs)

        records: List[RunRecord] = []
        total = len(specs)
        started = time.time()

        for index, spec in enumerate(specs, start=1):
            context = {"index": index, "total": total, "spec": spec}

            if self.resume and self.already_done(spec):
                self.on_event("skipped", context)
                continue

            self.on_event("start", context)
            record = self.run_one(spec)
            records.append(record)
            self._save(spec, record)
            self.on_event("finish", {**context, "record": record})

        self.on_event("done", {
            "total": total, "executed": len(records),
            "failed": sum(1 for r in records if not r.ok),
            "elapsed_s": time.time() - started,
        })
        return records

    def run_one(self, spec: RunSpec) -> RunRecord:
        """One spec to one record. Never raises; failures become failed records."""
        try:
            device = self.device(spec.device_id)
            backend = self.backend(spec.backend, spec.device_id)
        except Exception as exc:
            return self._failure(spec, f"{type(exc).__name__}: {exc}")

        try:
            task = backend.build_task(spec, device)
        except Exception as exc:
            return self._failure(spec, f"could not build task: {exc}")

        try:
            result = device.execute(task)
        except Exception as exc:
            return self._failure(spec, f"{type(exc).__name__}: {exc}")

        try:
            record = backend.parse(spec, result, device)
        except Exception as exc:
            # The run happened; only parsing broke. Keep the raw output so it can be
            # reparsed later rather than re-measured.
            record = self._failure(spec, f"could not parse result: {exc}")
            record.raw = {"stdout": (result.get("stdout") or "")[:200000],
                          "stderr": (result.get("stderr") or "")[-8000:]}
        return record

    def _failure(self, spec: RunSpec, message: str) -> RunRecord:
        from . import __version__
        from .schema import DeviceInfo, Knobs, Workload

        return RunRecord(
            run_id=compute_run_id(spec.identity()),
            experiment=spec.experiment,
            lab_version=__version__,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            status="failed",
            error=message,
            tags=list(spec.tags),
            device=DeviceInfo(device_id=spec.device_id),
            workload=Workload(model=spec.model, model_path=spec.model_path,
                              quantization=spec.quantization,
                              prompt_tokens=spec.prompt_tokens,
                              output_tokens=spec.output_tokens),
            knobs=Knobs(threads=spec.threads, governor=spec.governor,
                        freq_khz=spec.freq_khz, repetitions=spec.repetitions),
        )

    # ----------------------------------------------------------------------- output

    def _save(self, spec: RunSpec, record: RunRecord) -> str:
        path = self.record_path(spec)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write then rename, so a run interrupted mid-write cannot leave a half-parsed
        # file that poisons the next analysis pass.
        temporary = path + ".partial"
        with open(temporary, "w", encoding="utf-8") as fh:
            fh.write(record.to_json())
        os.replace(temporary, path)

        with open(os.path.join(self.output_dir, "index.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "run_id": record.run_id, "device": record.device.device_id,
                "backend": record.backend.name, "model": record.workload.model,
                "mode": spec.mode, "status": record.status,
                "path": os.path.relpath(path, self.output_dir),
            }) + "\n")
        return path

    def _write_manifest(self, specs: List[RunSpec]) -> None:
        """Record what was asked for, next to what came back."""
        manifest = {
            "experiment": self.config.experiment,
            "source_config": self.config.source_path,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_count": len(specs),
            "devices": self.config.devices,
            "models": self.config.models,
            "config": self.config.raw,
        }
        with open(os.path.join(self.output_dir, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, default=str)


def console_reporter(verbose: bool = True) -> Callable[[str, Dict[str, Any]], None]:
    """Progress printer for the CLI, kept out of Runner so Runner stays importable."""

    def report(kind: str, payload: Dict[str, Any]) -> None:
        spec = payload.get("spec")
        if kind == "start" and verbose:
            label = _label(spec)
            sys.stderr.write(f"[{payload['index']:>3}/{payload['total']}] {label} ... ")
            sys.stderr.flush()
        elif kind == "skipped" and verbose:
            sys.stderr.write(f"[{payload['index']:>3}/{payload['total']}] {_label(spec)} "
                             "... already done\n")
        elif kind == "finish":
            record = payload["record"]
            if not record.ok:
                sys.stderr.write(f"FAILED: {record.error}\n")
            else:
                sys.stderr.write(_metrics_line(record) + "\n")
            for warning in record.warnings:
                sys.stderr.write(f"        warning: {warning}\n")
        elif kind == "done":
            sys.stderr.write(
                f"\n{payload['executed']} run(s) in {payload['elapsed_s'] / 60:.1f} min, "
                f"{payload['failed']} failed\n"
            )

    return report


def _label(spec: RunSpec) -> str:
    bits = [spec.device_id, spec.model, spec.mode]
    if spec.threads:
        bits.append(f"t{spec.threads}")
    bits.append(f"p{spec.prompt_tokens}/n{spec.output_tokens}")
    return " ".join(bits)


def _metrics_line(record: RunRecord) -> str:
    m = record.metrics
    parts = []
    if m.ttft_ms is not None:
        parts.append(f"ttft {m.ttft_ms:.0f} ms")
    if m.prefill_tps is not None:
        parts.append(f"prefill {m.prefill_tps:.1f} t/s")
    if m.decode_tps is not None:
        parts.append(f"decode {m.decode_tps:.2f} t/s")
    if m.latency_ms_mean is not None:
        parts.append(f"latency {m.latency_ms_mean:.2f} ms")
    if m.bw_utilization_pct is not None:
        parts.append(f"bw {m.bw_utilization_pct:.0f}%")
    if record.telemetry.power_w_mean is not None:
        parts.append(f"{record.telemetry.power_w_mean:.2f} W")
    if record.telemetry.temp_c_max is not None:
        parts.append(f"{record.telemetry.temp_c_max:.0f} C")
    return ", ".join(parts) if parts else "ok"
