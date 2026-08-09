"""Device registry: turn a config block into something a backend can run on."""

from __future__ import annotations

from typing import Any, Dict

from .base import Device, DeviceError
from .local import LocalDevice
from .ssh import SSHDevice

_REGISTRY = {
    "local": LocalDevice,
    "ssh": SSHDevice,
}


def register(kind: str, cls) -> None:
    """Add a device kind. A future GPU box or a cloud runner plugs in here."""
    _REGISTRY[kind] = cls


def from_config(device_id: str, config: Dict[str, Any]) -> Device:
    """Build a device from its config block.

    The kind defaults to ssh when a host is given and local otherwise, so the common case
    in a config file is just ``host:`` and a key path.
    """
    config = dict(config or {})
    kind = config.get("kind") or ("ssh" if config.get("host") else "local")
    cls = _REGISTRY.get(kind)
    if cls is None:
        known = ", ".join(sorted(_REGISTRY))
        raise DeviceError(f"unknown device kind '{kind}' for '{device_id}'. Known: {known}")
    return cls(device_id, config)


__all__ = ["Device", "DeviceError", "LocalDevice", "SSHDevice", "from_config", "register"]
