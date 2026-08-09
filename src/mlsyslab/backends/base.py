"""What a backend is: something that turns one point of a sweep into one measurement.

The contract is deliberately one-to-one. A :class:`RunSpec` describes a single point, the
backend turns it into a single agent task, and parsing that task's output yields a single
:class:`~mlsyslab.schema.RunRecord`. Aggregation across points is the analysis layer's
job, not the backend's.

Parsing is separated from running on purpose. The agent returns raw stdout and the host
parses it, so a parser bug discovered in a month can be fixed by re-reading stored output
instead of by re-running a campaign that took a night.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..schema import BackendInfo, RunRecord


@dataclass
class RunSpec:
    """One point in a sweep: everything needed to produce exactly one measurement."""

    experiment: str = "unnamed"
    device_id: str = "local"
    backend: str = "llamacpp"
    mode: str = "throughput"          # throughput | latency
    model: str = "unknown"
    model_path: Optional[str] = None
    quantization: Optional[str] = None
    prompt_tokens: int = 128
    output_tokens: int = 128
    context_tokens: Optional[int] = None
    batch_size: Optional[int] = None
    threads: Optional[int] = None
    governor: Optional[str] = None
    freq_khz: Optional[int] = None
    repetitions: int = 3
    warmup: Optional[int] = None
    flash_attention: Optional[bool] = None
    kv_quantization: Optional[str] = None
    sample_power: bool = False        # off by default: sampling perturbs throughput
    sample_hz: float = 10.0
    prepare: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def identity(self) -> Dict[str, Any]:
        """The fields that make this point distinct, for a deterministic run id.

        Excludes anything incidental (tags, sampling rate) so that resuming a campaign
        recognises work it has already done.
        """
        return {
            "experiment": self.experiment, "device": self.device_id,
            "backend": self.backend, "mode": self.mode, "model": self.model,
            "quant": self.quantization, "prompt": self.prompt_tokens,
            "output": self.output_tokens, "context": self.context_tokens,
            "batch": self.batch_size, "threads": self.threads,
            "governor": self.governor, "freq_khz": self.freq_khz,
            "reps": self.repetitions, "fa": self.flash_attention,
            "kv_quant": self.kv_quantization, "power": self.sample_power,
        }


class BackendError(RuntimeError):
    pass


class Backend:
    """Base class for inference backends."""

    name = "base"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(config or {})
        self._info_cache: Dict[str, BackendInfo] = {}

    # ------------------------------------------------------------------ interface

    def discover(self, device) -> BackendInfo:
        """Find the backend on the device and report its version. Cached per device."""
        raise NotImplementedError

    def build_task(self, spec: RunSpec, device) -> Dict[str, Any]:
        """Turn one spec into one agent task."""
        raise NotImplementedError

    def parse(self, spec: RunSpec, result: Dict[str, Any], device) -> RunRecord:
        """Turn one agent result into one RunRecord."""
        raise NotImplementedError

    # -------------------------------------------------------------------- helpers

    def info(self, device) -> BackendInfo:
        if device.device_id not in self._info_cache:
            self._info_cache[device.device_id] = self.discover(device)
        return self._info_cache[device.device_id]

    def supports(self, spec: RunSpec) -> bool:
        return True

    def __repr__(self) -> str:
        return f"<Backend {self.name}>"
