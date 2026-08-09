"""One file in, a list of concrete runs out.

An experiment is reproducible from one command only if the sweep lives in a file rather
than in a shell loop. This module turns a declarative config into a fully expanded list
of :class:`~mlsyslab.backends.base.RunSpec`, so what will run can be inspected before a
single measurement is taken, and so a campaign can be resumed by run id.

YAML is the intended format; JSON is accepted so the package stays usable without PyYAML,
which matters for the dependency-free CI job.
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .backends.base import RunSpec


class ConfigError(ValueError):
    pass


# Axes that expand into a cartesian product. Anything else is a scalar setting that is
# applied to every spec unchanged.
_SWEEP_AXES = {
    "devices": "device_id",
    "models": "model",
    "modes": "mode",
    "threads": "threads",
    "prompt_tokens": "prompt_tokens",
    "output_tokens": "output_tokens",
    "context_tokens": "context_tokens",
    "batch_sizes": "batch_size",
    "governors": "governor",
    "freqs_khz": "freq_khz",
    "kv_quantizations": "kv_quantization",
}

_SCALARS = {
    "backend", "repetitions", "warmup", "flash_attention", "sample_power",
    "sample_hz", "prepare", "tags", "extra",
}


@dataclass
class Config:
    experiment: str = "unnamed"
    output_dir: str = "runs"
    devices: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    specs: List[RunSpec] = field(default_factory=list)
    source_path: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def device_ids(self) -> List[str]:
        seen: List[str] = []
        for spec in self.specs:
            if spec.device_id not in seen:
                seen.append(spec.device_id)
        return seen

    def summary(self) -> str:
        by_device: Dict[str, int] = {}
        for spec in self.specs:
            by_device[spec.device_id] = by_device.get(spec.device_id, 0) + 1
        parts = ", ".join(f"{d}: {n}" for d, n in sorted(by_device.items()))
        return f"{self.experiment}: {len(self.specs)} runs ({parts})"


def _load_document(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:
            raise ConfigError(
                "reading YAML configs needs PyYAML (pip install pyyaml), "
                "or write the config as JSON"
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return [None]
    if isinstance(value, (list, tuple)):
        return list(value) or [None]
    return [value]


def load(path: str) -> Config:
    """Read a config file and expand it into concrete run specs."""
    data = _load_document(path)

    devices = data.get("devices") or {}
    if not isinstance(devices, dict) or not devices:
        raise ConfigError(f"{path} must define at least one device under 'devices'")
    models = data.get("models") or {}

    config = Config(
        experiment=data.get("experiment") or os.path.splitext(os.path.basename(path))[0],
        output_dir=data.get("output_dir") or os.path.join("runs", str(data.get("experiment", "unnamed"))),
        devices=devices,
        models=models,
        source_path=os.path.abspath(path),
        raw=data,
    )

    defaults = data.get("defaults") or {}
    blocks = data.get("matrix")
    if not blocks:
        raise ConfigError(f"{path} has no 'matrix' section, so there is nothing to run")
    if isinstance(blocks, dict):
        blocks = [blocks]

    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise ConfigError(f"matrix entry {index} is not a mapping")
        merged = dict(defaults)
        merged.update(block)
        config.specs.extend(_expand_block(config, merged, index))

    _validate(config)
    return config


def _expand_block(config: Config, block: Dict[str, Any], index: int) -> List[RunSpec]:
    axis_values = []
    axis_fields = []
    for key, field_name in _SWEEP_AXES.items():
        if key in block:
            axis_values.append(_as_list(block[key]))
            axis_fields.append(field_name)

    if not axis_values:
        raise ConfigError(f"matrix entry {index} sweeps nothing; at least 'devices' is required")

    specs: List[RunSpec] = []
    for combination in itertools.product(*axis_values):
        assignment = dict(zip(axis_fields, combination))
        specs.append(_make_spec(config, block, assignment, index))
    return specs


def _make_spec(config: Config, block: Dict[str, Any],
               assignment: Dict[str, Any], index: int) -> RunSpec:
    device_id = assignment.get("device_id")
    if not device_id:
        raise ConfigError(f"matrix entry {index} has no device")
    if device_id not in config.devices:
        known = ", ".join(sorted(config.devices))
        raise ConfigError(f"matrix entry {index} names unknown device '{device_id}'. Known: {known}")

    spec = RunSpec(experiment=config.experiment, device_id=device_id)
    for field_name, value in assignment.items():
        if value is not None:
            setattr(spec, field_name, value)

    for key in _SCALARS:
        if key in block and block[key] is not None:
            attr = {"sample_power": "sample_power", "sample_hz": "sample_hz"}.get(key, key)
            if hasattr(spec, attr):
                setattr(spec, attr, block[key])

    spec.mode = spec.mode or "throughput"
    _attach_model(config, spec, index)

    # Defaults that only make sense once the mode is known. Latency runs measure a single
    # request, so a high repetition count is spread rather than precision; throughput runs
    # want llama-bench's own repetitions.
    if spec.mode == "latency" and "repetitions" not in block:
        spec.repetitions = 3
    return spec


def _attach_model(config: Config, spec: RunSpec, index: int) -> None:
    if spec.model in (None, "unknown"):
        return
    entry = config.models.get(spec.model)
    if entry is None:
        known = ", ".join(sorted(config.models)) or "none defined"
        raise ConfigError(
            f"matrix entry {index} names unknown model '{spec.model}'. Known: {known}"
        )
    if isinstance(entry, str):
        entry = {"path": entry}

    paths = entry.get("paths") or {}
    spec.model_path = paths.get(spec.device_id) or entry.get("path")
    if not spec.model_path:
        raise ConfigError(
            f"model '{spec.model}' has no path for device '{spec.device_id}'. "
            "Give it 'path' for every device, or 'paths' keyed by device id"
        )
    if entry.get("quantization"):
        spec.quantization = entry["quantization"]
    if entry.get("tags"):
        spec.tags = list(spec.tags) + list(entry["tags"])
    for key in ("batch_size", "input_shape"):
        if entry.get(key) is not None:
            spec.extra.setdefault(key, entry[key])


def _validate(config: Config) -> None:
    if not config.specs:
        raise ConfigError("config expanded to zero runs")
    ids: Dict[str, RunSpec] = {}
    from .schema import compute_run_id

    for spec in config.specs:
        run_id = compute_run_id(spec.identity())
        if run_id in ids:
            # Identical points are not an error worth stopping for, but they are wasted
            # machine time and they collide on disk, so say so.
            raise ConfigError(
                f"two runs in this config are identical ({spec.model} on {spec.device_id}, "
                f"mode {spec.mode}). Remove the duplicate axis value."
            )
        ids[run_id] = spec


def specs_to_table(specs: Iterable[RunSpec]) -> str:
    """A human-readable preview of what will run, for --dry-run.

    Token columns appear only when something in the matrix actually consumes tokens.
    Printing "prompt 128" next to an image classifier is not a harmless default; it is a
    column of numbers that were never measured and do not apply.
    """
    specs = list(specs)
    token_based = any(s.backend not in ("onnxruntime",) for s in specs)

    header = ["device", "backend", "mode", "model", "threads"]
    header += ["prompt", "output"] if token_based else ["batch"]
    rows = [tuple(header)]
    for spec in specs:
        row = [spec.device_id, spec.backend, spec.mode, spec.model, str(spec.threads or "-")]
        if token_based:
            row += ([str(spec.prompt_tokens), str(spec.output_tokens)]
                    if spec.backend != "onnxruntime" else ["-", "-"])
        else:
            row += [str(spec.batch_size or 1)]
        rows.append(tuple(row))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for position, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
        if position == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)
