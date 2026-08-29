"""Device registry: turn a config block into something a backend can run on."""

from __future__ import annotations

from typing import Any, Dict

from .base import Device, DeviceError
from .k8s import K8sDevice
from .local import LocalDevice
from .ssh import SSHDevice

_REGISTRY = {
    "local": LocalDevice,
    "ssh": SSHDevice,
    "k8s": K8sDevice,
}


def register(kind: str, cls) -> None:
    """Add a device kind. A future GPU box or a cloud runner plugs in here."""
    _REGISTRY[kind] = cls


def from_config(device_id: str, config: Dict[str, Any]) -> Device:
    """Build a device from its config block.

    The kind is inferred from what the block contains, so the common case in a config
    file is a couple of lines: a ``host`` means SSH, a ``pod`` or an ``image`` means
    Kubernetes, and neither means this machine. An explicit ``kind`` always wins.
    """
    config = dict(config or {})
    # An explicit kind always wins. Otherwise a pod or an image means Kubernetes and a
    # host means SSH, so the common case in a config file stays a couple of lines.
    kind = config.get("kind")
    if not kind:
        if config.get("pod") or config.get("image") or config.get("namespace"):
            kind = "k8s"
        elif config.get("host"):
            kind = "ssh"
        else:
            kind = "local"
    cls = _REGISTRY.get(kind)
    if cls is None:
        known = ", ".join(sorted(_REGISTRY))
        raise DeviceError(f"unknown device kind '{kind}' for '{device_id}'. Known: {known}")
    return cls(device_id, config)


__all__ = ["Device", "DeviceError", "K8sDevice", "LocalDevice", "SSHDevice",
           "from_config", "register"]
