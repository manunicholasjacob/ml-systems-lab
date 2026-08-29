"""A device reached over SSH, such as the Raspberry Pi on the tailnet.

The package is copied to the device once per session and the agent is invoked there, so
the benchmark and its telemetry sampler are colocated. Nothing is installed: the copy
lands in a working directory and runs from ``PYTHONPATH``, because the target is a 2 GB
board where a pip install is an event rather than a detail.

One SSH invocation per run, rather than one long-lived session for the whole campaign.
A dropped connection then costs a single point instead of an eight hour matrix, which is
worth more than the second or so of connection setup per run.
"""

from __future__ import annotations

import os
import posixpath
import subprocess
from typing import Any, Dict, List, Optional

from .base import Device, DeviceError

# Keep a long campaign alive through idle stretches, and fail fast when the tailnet drops
# rather than blocking a run forever on a dead socket.
_DEFAULT_SSH_OPTIONS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=20",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=6",
]


def _first_existing_script(paths):
    """One shell command that names the first path that exists, or says none did."""
    quoted = " ".join(_sq(p) for p in paths)
    return (
        "for p in " + quoted + "; do "
        'if [ -e "$p" ]; then printf \'MLSYS_HIT:%s\' "$p"; exit 0; fi; '
        "done; printf 'MLSYS_NONE'"
    )


def _parse_first_existing(text):
    """None means nothing matched; a raised ValueError means we could not tell."""
    text = text.strip()
    marker = text.rfind("MLSYS_HIT:")
    if marker != -1:
        return text[marker + len("MLSYS_HIT:"):].strip()
    if text.endswith("MLSYS_NONE"):
        return None
    raise ValueError(text[-300:] or "no output")


def _sq(text: str) -> str:
    """Single-quote a string for /bin/sh."""
    return "'" + text.replace("'", "'\\''") + "'"


class SSHDevice(Device):
    kind = "ssh"

    def __init__(self, device_id: str, config: Dict[str, Any]):
        super().__init__(device_id, config)
        if not config.get("host"):
            raise DeviceError(f"device '{device_id}' is kind ssh but has no host")
        self.host = config["host"]
        self.user = config.get("user")
        self.port = config.get("port")
        self.identity_file = config.get("identity_file")
        self.python = config.get("python", "python3")
        self.workdir = config.get("workdir", "~/.mlsyslab")
        self.extra_options: List[str] = list(config.get("ssh_options") or [])
        self._synced = False
        self._home: Optional[str] = None

    # -------------------------------------------------------------------- plumbing

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    def _ssh_base(self) -> List[str]:
        cmd = ["ssh"] + _DEFAULT_SSH_OPTIONS
        if self.identity_file:
            cmd += ["-i", os.path.expanduser(self.identity_file)]
        if self.port:
            cmd += ["-p", str(self.port)]
        return cmd + self.extra_options + [self.target]

    def _scp_base(self) -> List[str]:
        cmd = ["scp"] + _DEFAULT_SSH_OPTIONS
        if self.identity_file:
            cmd += ["-i", os.path.expanduser(self.identity_file)]
        if self.port:
            cmd += ["-P", str(self.port)]
        return cmd + self.extra_options

    def run_shell(self, script: str, timeout_s: float = 120,
                  stdin: Optional[bytes] = None) -> subprocess.CompletedProcess:
        """Run a shell snippet on the device. Used for setup, never for measurement."""
        try:
            return subprocess.run(
                self._ssh_base() + [script],
                input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=None if stdin is not None else subprocess.DEVNULL,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise DeviceError(f"ssh to {self.target} timed out after {timeout_s:.0f} s") from exc
        except OSError as exc:
            raise DeviceError(f"could not run ssh: {exc}") from exc

    # ----------------------------------------------------------------------- paths

    def _remote_home(self) -> str:
        if self._home is None:
            done = self.run_shell("printf %s \"$HOME\"", timeout_s=60)
            home = done.stdout.decode("utf-8", "replace").strip()
            if done.returncode != 0 or not home:
                raise DeviceError(f"could not determine home directory on {self.target}")
            self._home = home
        return self._home

    def resolve(self, path: str) -> str:
        """Expand ``~`` the way the remote shell would.

        Done locally against a cached home directory rather than by asking the device
        each time. The agent invokes binaries without a shell, so an unexpanded ``~``
        would reach ``execvp`` verbatim and fail with a confusing "no such file".
        """
        if path.startswith("~"):
            return posixpath.normpath(self._remote_home() + path[1:])
        return path

    def exists(self, path: str) -> bool:
        """Whether the path is on the device, or a loud failure if we could not look.

        ``ssh`` exits non-zero both when ``test -e`` says no and when the connection
        dropped, and a benchmark that cannot tell those apart records a dropped tailnet
        as a missing model file. The answer is printed and insisted upon instead.
        """
        quoted = self.resolve(path).replace("'", "'\\''")
        done = self.run_shell(
            f"if [ -e '{quoted}' ]; then echo MLSYS_YES; else echo MLSYS_NO; fi",
            timeout_s=60,
        )
        answer = done.stdout.decode("utf-8", "replace").strip()
        if answer.endswith("MLSYS_YES"):
            return True
        if answer.endswith("MLSYS_NO"):
            return False
        raise DeviceError(
            f"could not determine whether {path} exists on {self.target} "
            f"(exit {done.returncode}): "
            + (done.stderr.decode("utf-8", "replace")[-300:] or "no output")
        )

    def first_existing(self, paths: List[str]) -> Optional[str]:
        """Ask about every candidate over one connection instead of one each.

        Backend discovery searches eight directories for two names. One path per
        connection is sixteen SSH handshakes per run, which is both slow and enough
        churn to trip a remote sshd's startup rate limiting. When that happened the
        dropped connections came back as "the binary is not there", on a machine where
        it was.
        """
        if not paths:
            return None
        done = self.run_shell(_first_existing_script(paths), timeout_s=90)
        try:
            return _parse_first_existing(done.stdout.decode("utf-8", "replace"))
        except ValueError as exc:
            raise DeviceError(
                f"could not search {self.target} for {len(paths)} candidate path(s) "
                f"(exit {done.returncode}): {exc}"
            ) from exc

    # ------------------------------------------------------------------ deployment

    def sync(self, force: bool = False) -> str:
        """Copy this package to the device. Returns the remote package root."""
        remote_root = self.resolve(self.workdir)
        if self._synced and not force:
            return remote_root

        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        done = self.run_shell(f"mkdir -p '{remote_root}' && rm -rf '{remote_root}/mlsyslab'")
        if done.returncode != 0:
            raise DeviceError(
                f"could not prepare {remote_root} on {self.target}: "
                + done.stderr.decode("utf-8", "replace")[-500:]
            )

        scp = self._scp_base() + ["-r", package_dir, f"{self.target}:{remote_root}/"]
        try:
            copied = subprocess.run(scp, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    stdin=subprocess.DEVNULL, timeout=300)
        except (OSError, subprocess.SubprocessError) as exc:
            raise DeviceError(f"scp to {self.target} failed: {exc}") from exc
        if copied.returncode != 0:
            raise DeviceError(
                "scp failed: " + copied.stderr.decode("utf-8", "replace")[-500:]
            )

        # Byte-compiled files from the host are for the wrong interpreter and confuse
        # nothing but a reader; drop them so the remote tree is exactly the source.
        self.run_shell(f"find '{remote_root}/mlsyslab' -name __pycache__ -type d "
                       f"-exec rm -rf {{}} + 2>/dev/null; true")
        self._synced = True
        return remote_root

    def package_root(self) -> str:
        return self.sync()

    @property
    def python_executable(self) -> str:
        return self.python

    def _invoke_agent(self, task_json: str, timeout_s: float) -> str:
        remote_root = self.sync()
        command = f"cd '{remote_root}' && PYTHONPATH='{remote_root}' {self.python} -m mlsyslab.agent -"
        done = self.run_shell(command, timeout_s=timeout_s, stdin=task_json.encode("utf-8"))
        stdout = done.stdout.decode("utf-8", "replace")
        if done.returncode != 0 and not stdout.strip():
            stderr = done.stderr.decode("utf-8", "replace")[-2000:]
            raise DeviceError(f"remote agent failed (exit {done.returncode}):\n{stderr}")
        return stdout

    # ----------------------------------------------------------------- file moves

    def push(self, local_path: str, remote_path: str) -> None:
        flags = ["-r"] if os.path.isdir(local_path) else []
        scp = self._scp_base() + flags + [local_path, f"{self.target}:{remote_path}"]
        done = subprocess.run(scp, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              stdin=subprocess.DEVNULL, timeout=1800)
        if done.returncode != 0:
            raise DeviceError("push failed: " + done.stderr.decode("utf-8", "replace")[-500:])

    def pull(self, remote_path: str, local_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(local_path)) or ".", exist_ok=True)
        scp = self._scp_base() + ["-r", f"{self.target}:{remote_path}", local_path]
        done = subprocess.run(scp, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              stdin=subprocess.DEVNULL, timeout=1800)
        if done.returncode != 0:
            raise DeviceError("pull failed: " + done.stderr.decode("utf-8", "replace")[-500:])
