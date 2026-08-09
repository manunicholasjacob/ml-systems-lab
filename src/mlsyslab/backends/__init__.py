"""Backend registry."""

from __future__ import annotations

from typing import Any, Dict

from .base import Backend, BackendError, RunSpec

_REGISTRY: Dict[str, type] = {}


def register(name: str, cls: type) -> None:
    _REGISTRY[name] = cls


def get(name: str, config: Dict[str, Any] = None) -> Backend:
    cls = _REGISTRY.get(name)
    if cls is None:
        known = ", ".join(sorted(_REGISTRY)) or "none registered"
        raise BackendError(f"unknown backend '{name}'. Known: {known}")
    return cls(config)


def available() -> Dict[str, type]:
    return dict(_REGISTRY)


def _register_builtins() -> None:
    from .llamacpp import LlamaCppBackend

    register(LlamaCppBackend.name, LlamaCppBackend)
    # ONNX Runtime is registered lazily: importing it must not require onnxruntime on the
    # host, since the host may only be orchestrating a device that has it.
    try:
        from .onnxrt import OnnxRuntimeBackend

        register(OnnxRuntimeBackend.name, OnnxRuntimeBackend)
    except ImportError:
        pass


_register_builtins()

__all__ = ["Backend", "BackendError", "RunSpec", "get", "register", "available"]
