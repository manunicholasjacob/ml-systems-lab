"""This machine.

The agent runs in a subprocess rather than in-process, even though in-process would be a
function call. Two reasons, both of which showed up in practice:

* Peak RSS is read from ``getrusage(RUSAGE_CHILDREN)``, which accumulates over the life
  of the process. In-process, the second run in a sweep would inherit the first run's
  peak and every number after it would be wrong in the same direction.
* A backend that segfaults or wedges takes the subprocess with it, not the campaign.

It also means local and SSH devices behave identically, which is the point of having the
abstraction at all.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Dict, Optional

from .base import Device, DeviceError


class LocalDevice(Device):
    kind = "local"

    def __init__(self, device_id: str = "local", config: Optional[Dict[str, Any]] = None):
        super().__init__(device_id, config)
        self.python = self.config.get("python") or sys.executable

    def _invoke_agent(self, task_json: str, timeout_s: float) -> str:
        env = dict(os.environ)
        # Run the agent from this checkout even when the package is not installed.
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src_root = os.path.dirname(package_root)
        env["PYTHONPATH"] = os.pathsep.join(
            [src_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        try:
            done = subprocess.run(
                [self.python, "-m", "mlsyslab.agent", "-"],
                input=task_json.encode("utf-8"),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=timeout_s, env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise DeviceError(f"local agent exceeded {timeout_s:.0f} s") from exc
        except OSError as exc:
            raise DeviceError(f"could not start local agent: {exc}") from exc

        if done.returncode != 0 and not done.stdout:
            stderr = done.stderr.decode("utf-8", "replace")[-2000:]
            raise DeviceError(f"local agent failed (exit {done.returncode}):\n{stderr}")
        return done.stdout.decode("utf-8", "replace")

    def push(self, local_path: str, remote_path: str) -> None:
        import shutil

        if os.path.abspath(local_path) == os.path.abspath(remote_path):
            return
        if os.path.isdir(local_path):
            shutil.copytree(local_path, remote_path, dirs_exist_ok=True)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(remote_path)), exist_ok=True)
            shutil.copy2(local_path, remote_path)

    def pull(self, remote_path: str, local_path: str) -> None:
        self.push(remote_path, local_path)

    def resolve(self, path: str) -> str:
        return os.path.expandvars(os.path.expanduser(path))

    def exists(self, path: str) -> bool:
        return os.path.exists(self.resolve(path))

    def package_root(self) -> str:
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.dirname(package_dir)

    @property
    def python_executable(self) -> str:
        return self.python
