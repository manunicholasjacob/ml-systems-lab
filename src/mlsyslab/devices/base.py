"""What a device is: somewhere a task can run and telemetry can be collected.

The abstraction is deliberately narrow. A device does not know what a benchmark is, what
a model is, or how to parse anything. It knows how to run the agent and give back the
agent's result. That is what lets the same backend code target a laptop, a Pi over SSH,
and later a server, without a branch anywhere in the backend.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..agent import RESULT_BEGIN, RESULT_END


class DeviceError(RuntimeError):
    pass


class Device:
    """Base class. Subclasses implement :meth:`_invoke_agent` and :meth:`push`."""

    kind = "base"

    def __init__(self, device_id: str, config: Optional[Dict[str, Any]] = None):
        self.device_id = device_id
        self.config = dict(config or {})
        self._probe_cache: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ interface

    def _invoke_agent(self, task_json: str, timeout_s: float) -> str:
        raise NotImplementedError

    def push(self, local_path: str, remote_path: str) -> None:
        raise NotImplementedError

    def pull(self, remote_path: str, local_path: str) -> None:
        raise NotImplementedError

    def resolve(self, path: str) -> str:
        """Expand a path the way the *device* would, not the way the host would."""
        return path

    def exists(self, path: str) -> bool:
        """Whether the path is there.

        Implementations must raise :class:`DeviceError` rather than returning False when
        they could not find out. The two are not the same answer, and conflating them is
        how a dropped connection during discovery turns into "llama-bench not found on
        k8s-cpu2", which reads as a configuration mistake, is recorded as one, and is
        therefore never retried.
        """
        raise NotImplementedError

    def package_root(self) -> str:
        """Directory on the device from which ``mlsyslab`` is importable.

        Needed by backends whose measurement path is a Python module rather than a
        binary, so they can set PYTHONPATH for the subprocess the agent launches.
        """
        raise NotImplementedError

    @property
    def python_executable(self) -> str:
        """Interpreter to use for Python-based backends on this device.

        Frequently not the default one: the laptop's ``python`` is 3.14 with no ONNX
        Runtime, so its config points at a 3.11 virtual environment instead.
        """
        return self.config.get("python") or "python3"

    # -------------------------------------------------------------------- running

    def execute(self, task: Dict[str, Any], timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """Run one agent task on this device and return the parsed result."""
        task = dict(task)
        task.setdefault("device_id", self.device_id)
        task.setdefault("device_kind", self.kind)
        # Give the transport headroom over the task's own timeout, so a task that times
        # out reports its own partial result instead of being killed by the transport.
        budget = timeout_s or (float(task.get("timeout_s") or 3600) + 300)
        raw = self._invoke_agent(json.dumps(task), budget)
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> Dict[str, Any]:
        """Pull the result out from between the sentinels.

        SSH mixes banners, sudo lectures and library warnings into the same stream, so the
        payload is fenced rather than assumed to be the whole of stdout.
        """
        start = raw.find(RESULT_BEGIN)
        end = raw.find(RESULT_END)
        if start == -1 or end == -1 or end < start:
            tail = raw[-2000:] if raw else "(no output)"
            raise DeviceError(f"agent produced no parseable result. Output tail:\n{tail}")
        blob = raw[start + len(RESULT_BEGIN):end].strip()
        try:
            return json.loads(blob)
        except ValueError as exc:
            raise DeviceError(f"agent result was not valid JSON: {exc}") from exc

    # -------------------------------------------------------------------- probing

    def probe(self, refresh: bool = False) -> Dict[str, Any]:
        """Hardware description and capability map, cached per device."""
        if self._probe_cache is None or refresh:
            result = self.execute({"kind": "sysinfo"}, timeout_s=180)
            self._probe_cache = {
                "device": result.get("device", {}),
                "capabilities": result.get("capabilities", {}),
            }
        return self._probe_cache

    def capabilities(self) -> Dict[str, bool]:
        return dict(self.probe().get("capabilities", {}))

    def can(self, capability: str) -> bool:
        return bool(self.capabilities().get(capability))

    def device_info(self) -> Dict[str, Any]:
        info = dict(self.probe().get("device", {}))
        info["device_id"] = self.device_id
        info["kind"] = self.kind
        info["capabilities"] = self.capabilities()
        # Peak DRAM bandwidth cannot be probed reliably, so it is declared in the device
        # config and carried through. Roofline utilisation is meaningless without it.
        for key in ("dram_peak_GBs", "dram_measured_GBs", "accelerator"):
            if self.config.get(key) is not None:
                info[key] = self.config[key]
        return info

    # --------------------------------------------------------------------- lookup

    def find_binary(self, names: List[str], search_roots: List[str]) -> Optional[str]:
        """First existing path formed from the given names and roots, or None."""
        candidates = [
            self.resolve(f"{root.rstrip('/')}/{name}" if root else name)
            for root in search_roots for name in names
        ]
        return self.first_existing(candidates)

    def first_existing(self, paths: List[str]) -> Optional[str]:
        """The first path here that exists, or None if none of them do.

        A separate method from :meth:`exists` because asking about sixteen paths one at a
        time costs sixteen round trips, and on a device reached over SSH that is sixteen
        connections per run for a question with one answer. It was enough connection
        churn to trip the remote sshd's rate limiting, which then surfaced as
        "llama-bench not found" on a machine that had it. Devices that can ask once
        override this; the default is the honest loop.
        """
        for path in paths:
            if self.exists(path):
                return path
        return None

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.device_id}>"
