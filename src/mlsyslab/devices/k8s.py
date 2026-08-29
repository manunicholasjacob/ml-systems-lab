"""A device that is a container in a Kubernetes pod.

This is the third implementation of the same narrow interface as
:class:`~mlsyslab.devices.local.LocalDevice` and :class:`~mlsyslab.devices.ssh.SSHDevice`,
and it exists to answer one question honestly: what does orchestration cost a
measurement? Nothing in ``backends/`` knows this class exists, and if a backend ever
needs a branch for it then the abstraction has been broken and this file is wrong.

Three decisions here are not the obvious ones, and each came from something that broke.

**kubectl is an argv prefix, not a binary path.** The cluster this was developed against
is k3s inside WSL2, whose API server is not reachable from the Windows side because the
NAT does not forward 6443. Making ``kubectl`` a list means the config can say
``["wsl", "-d", "Ubuntu", "--exec", "k3s", "kubectl"]`` and the device does not care. The
same key handles a kubeconfig on a jump host, or ``kubectl`` under a different name. A
single string is accepted too, for the ordinary case.

The ``--exec`` in that example is not decoration. ``wsl -d Ubuntu -- <argv>`` joins the
arguments and hands the result to a login shell, which re-parses them: quotes vanish and
``$HOME`` is expanded on the *WSL host* rather than in the container. That does not fail,
it answers the wrong question, which is why :meth:`K8sDevice.verify_transport` exists and
runs before the first measurement. Any prefix that re-parses its arguments is rejected
with an explanation instead of being allowed to produce plausible numbers.

**Cluster state is read as JSON, never as jsonpath.** ``-o jsonpath=...`` needs quotes,
parentheses and question marks to survive the transport intact. Through the prefix above
they did not, and ``kubectl`` reported a pod as not Ready while it was serving. Asking for
the whole object and picking fields apart in Python has no characters that anything can
chew on.

**Files move as a tar stream over exec, not with ``kubectl cp``.** ``kubectl cp`` is
itself tar over exec, plus a host-path translation step. That translation is exactly what
breaks when the harness and kubectl see different filesystems, which is the situation
above: the harness holds a ``C:\\...`` path and kubectl inside WSL cannot open it. Doing
the tar ourselves removes the translation and the failure mode with it. The container
requirement is unchanged: ``kubectl cp`` also needs ``tar`` in the image.

**A sync lasts one container lifetime, not one session.** The SSH device copies the
package to the Pi once and the Pi keeps it, because a filesystem is a filesystem. A
container's is not: when the kubelet restarts the container, everything written into it
since it started is gone, and the pod carries on reporting Running the whole time. The
symptom is a sweep that works for twenty minutes and then reports
``No module named 'mlsyslab'`` on every device at once. So the sync is keyed to the
container's identity, and a container that restarted is re-synced before it is used.

**Truncation is named, not inferred.** ``kubectl exec`` can close a stream early under
load, and a half-delivered payload that still parses as JSON would be a silently wrong
measurement. A result whose opening sentinel arrived without its closing one is reported
as a truncated stream rather than parsed, because a missing run is recoverable and a
wrong number is not.
"""

from __future__ import annotations

import io
import json
import os
import posixpath
import subprocess
import tarfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from ..agent import RESULT_BEGIN, RESULT_END
from .base import Device, DeviceError

# Long enough that a slow image pull on first use is not mistaken for a broken cluster,
# short enough that an unschedulable pod fails the run instead of hanging the campaign.
_DEFAULT_READY_TIMEOUT_S = 600.0

# A running pod carries the hash of the spec that created it. Changing cpu_limit in a
# config and reusing the pod that is already there would run the whole sweep under the
# old quota and report it under the new label, which is the exact shape of a silently
# wrong result this repository exists to avoid.
_SPEC_HASH = "mlsyslab/spec-hash"


class K8sDevice(Device):
    kind = "k8s"

    def __init__(self, device_id: str, config: Dict[str, Any]):
        super().__init__(device_id, config)
        self.namespace = config.get("namespace") or "mlsyslab"
        self.pod = config.get("pod") or f"mlsyslab-{device_id}"
        self.container = config.get("container") or "agent"
        self.image = config.get("image")
        self.workdir = config.get("workdir") or "/opt/mlsyslab"
        self.python = config.get("python") or "python3"
        self.create_pod = config.get("create_pod", True)
        self.ready_timeout_s = float(config.get("ready_timeout_s") or _DEFAULT_READY_TIMEOUT_S)

        prefix = config.get("kubectl") or ["kubectl"]
        self.kubectl_prefix: List[str] = [prefix] if isinstance(prefix, str) else list(prefix)

        # sync() and ensure_pod() both mutate shared remote state, and the concurrent
        # scheduler can call into one device from several worker threads at once. Without
        # this, a second sync would delete the package tree out from under a running agent.
        self._lock = threading.RLock()
        self._synced = False
        self._synced_container: Optional[Tuple[Any, Any]] = None
        self._pod_ready = False
        self._transport_ok = False
        self._home: Optional[str] = None
        self._cgroup: Optional[Dict[str, Any]] = None

    # -------------------------------------------------------------------- plumbing

    def _kubectl(self, args: List[str], namespaced: bool = True) -> List[str]:
        cmd = list(self.kubectl_prefix)
        if self.config.get("kubeconfig"):
            cmd += ["--kubeconfig", str(self.config["kubeconfig"])]
        if self.config.get("context"):
            cmd += ["--context", str(self.config["context"])]
        if namespaced:
            cmd += ["-n", self.namespace]
        return cmd + args

    def _run(self, args: List[str], timeout_s: float = 120,
             stdin: Optional[bytes] = None,
             namespaced: bool = True) -> subprocess.CompletedProcess:
        cmd = self._kubectl(args, namespaced=namespaced)
        try:
            return subprocess.run(
                cmd, input=stdin,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=None if stdin is not None else subprocess.DEVNULL,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise DeviceError(
                f"kubectl {' '.join(args[:2])} on {self.namespace}/{self.pod} "
                f"timed out after {timeout_s:.0f} s"
            ) from exc
        except OSError as exc:
            raise DeviceError(f"could not run kubectl ({cmd[0]}): {exc}") from exc

    def _exec(self, argv: List[str], timeout_s: float = 120,
              stdin: Optional[bytes] = None) -> subprocess.CompletedProcess:
        """Run argv inside the container.

        No TTY is requested. With ``-t`` kubectl merges stderr into stdout and translates
        newlines, either of which would corrupt a fenced JSON payload or a tar stream.
        """
        args = ["exec"]
        if stdin is not None:
            args.append("-i")
        args += [self.pod, "-c", self.container, "--"] + argv
        return self._run(args, timeout_s=timeout_s, stdin=stdin)

    def _sh(self, script: str, timeout_s: float = 120) -> subprocess.CompletedProcess:
        return self._exec(["sh", "-c", script], timeout_s=timeout_s)

    # ------------------------------------------------------------- pod lifecycle

    def pod_manifest(self) -> Dict[str, Any]:
        """The pod this device runs in, as a dict.

        Built here rather than read from a YAML file so the resource limits that change
        the measurement are the same object that gets recorded in the run manifest. A
        limit that lives only in a file someone edited is a limit that will eventually
        disagree with the number it produced.
        """
        if not self.image:
            raise DeviceError(
                f"device '{self.device_id}' needs an 'image' to create pod "
                f"{self.namespace}/{self.pod}, or set create_pod: false and start it yourself"
            )

        container: Dict[str, Any] = {
            "name": self.container,
            "image": self.image,
            "imagePullPolicy": self.config.get("image_pull_policy") or "IfNotPresent",
            # The pod is a place to exec into, not a job, so it has to outlive many
            # runs. This was a shell loop with a SIGTERM trap, on the usual reasoning
            # that PID 1 ignores signals whose disposition is the default. That loop
            # exited by itself, seconds after starting, with status 0 and no output, and
            # took the harness's unpacked copy of itself with it every time. `sleep`
            # cannot do that: there is no shell, no job control and no trap to misfire.
            # The signal reasoning turns out not to apply either, because the kubelet
            # sends SIGTERM from outside the pod's PID namespace, and the kernel delivers
            # those to PID 1 normally.
            "command": ["sleep", "infinity"],
            "workingDir": self.workdir,
        }

        limits, requests = {}, {}
        for key, target in (("cpu_limit", "cpu"), ("memory_limit", "memory")):
            if self.config.get(key) is not None:
                limits[target] = str(self.config[key])
        for key, target in (("cpu_request", "cpu"), ("memory_request", "memory")):
            if self.config.get(key) is not None:
                requests[target] = str(self.config[key])
        if limits or requests:
            container["resources"] = {}
            if limits:
                container["resources"]["limits"] = limits
            if requests:
                container["resources"]["requests"] = requests

        mounts, volumes = [], []
        for index, entry in enumerate(self.config.get("host_mounts") or []):
            name = entry.get("name") or f"hostmount-{index}"
            mounts.append({"name": name, "mountPath": entry["mount_path"],
                           "readOnly": bool(entry.get("read_only", True))})
            volumes.append({"name": name,
                            "hostPath": {"path": entry["host_path"],
                                         "type": entry.get("type") or "Directory"}})
        if mounts:
            container["volumeMounts"] = mounts

        spec: Dict[str, Any] = {
            "containers": [container],
            "restartPolicy": "Always",
            # `sleep` dies immediately on SIGTERM, so there is nothing to wait for and a
            # pod delete should not sit through the default thirty seconds.
            "terminationGracePeriodSeconds": 5,
        }
        if volumes:
            spec["volumes"] = volumes
        if self.config.get("node_selector"):
            spec["nodeSelector"] = dict(self.config["node_selector"])
        if self.config.get("node_name"):
            spec["nodeName"] = str(self.config["node_name"])
        if self.config.get("tolerations"):
            spec["tolerations"] = list(self.config["tolerations"])
        if self.config.get("runtime_class"):
            spec["runtimeClassName"] = str(self.config["runtime_class"])

        manifest = {
            "apiVersion": "v1", "kind": "Pod",
            "metadata": {
                "name": self.pod, "namespace": self.namespace,
                "labels": {"app": "mlsyslab", "mlsyslab/device": self.device_id},
                "annotations": {},
            },
            "spec": spec,
        }
        manifest["metadata"]["annotations"][_SPEC_HASH] = _hash_spec(spec)
        return manifest

    def pod_json(self) -> Optional[Dict[str, Any]]:
        """The pod object, or None if it does not exist or could not be read."""
        done = self._run(["get", "pod", self.pod, "-o", "json"], timeout_s=90)
        if done.returncode != 0:
            return None
        try:
            payload = json.loads(done.stdout.decode("utf-8", "replace"))
        except ValueError:
            return None
        return payload if isinstance(payload, dict) else None

    def pod_phase(self) -> Optional[str]:
        pod = self.pod_json()
        return (pod or {}).get("status", {}).get("phase")

    def ensure_pod(self, force: bool = False) -> None:
        """Make the pod exist and be Ready. Idempotent, and safe to call from any thread."""
        with self._lock:
            if self._pod_ready and not force:
                return

            existing = self.pod_json()
            if existing is not None:
                # Staleness is checked whatever state the pod is in, not only when it is
                # Ready. A pod stuck Pending under yesterday's resource requests is
                # exactly the one that needs replacing, and skipping the check for it
                # sends an apply at an immutable spec, which fails with a wall of
                # diff and no explanation of what to do about it.
                stale = self._stale_reason(existing)
                if stale is None and self._is_ready(existing):
                    self._pod_ready = True
                    return
                if stale is not None:
                    # A pod whose spec no longer matches the config is worse than no
                    # pod: it would run happily and produce numbers under limits nobody
                    # asked for. Most of a pod spec is immutable, so the only honest fix
                    # is a new pod.
                    if not self.create_pod:
                        raise DeviceError(
                            f"pod {self.namespace}/{self.pod} does not match the config "
                            f"({stale}) and create_pod is false, so it cannot be "
                            "replaced. Delete it, or point the device at a pod that "
                            "matches."
                        )
                    self.delete_pod(wait=True)

            if not self.create_pod:
                raise DeviceError(
                    f"pod {self.namespace}/{self.pod} is not ready and create_pod is false"
                )

            self._run(["create", "namespace", self.namespace], timeout_s=60, namespaced=False)
            manifest = json.dumps(self.pod_manifest()).encode("utf-8")
            applied = self._run(["apply", "-f", "-"], timeout_s=120, stdin=manifest)
            if applied.returncode != 0:
                raise DeviceError(
                    f"could not apply pod {self.namespace}/{self.pod}: "
                    + applied.stderr.decode("utf-8", "replace")[-800:]
                )

            deadline = time.time() + self.ready_timeout_s
            while time.time() < deadline:
                pod = self.pod_json()
                if self._is_ready(pod):
                    self._pod_ready = True
                    # A pod that was recreated has an empty filesystem again.
                    self._synced = False
                    self._synced_container = None
                    self._home = None
                    self._cgroup = None
                    return
                phase = (pod or {}).get("status", {}).get("phase")
                if phase in ("Failed", "Succeeded"):
                    raise DeviceError(
                        f"pod {self.namespace}/{self.pod} reached phase {phase}: "
                        + self._describe_tail()
                    )
                time.sleep(2.0)

            raise DeviceError(
                f"pod {self.namespace}/{self.pod} was not ready within "
                f"{self.ready_timeout_s:.0f} s: " + self._describe_tail()
            )

    def _is_ready(self, pod: Optional[Dict[str, Any]] = None) -> bool:
        pod = pod if pod is not None else self.pod_json()
        if not pod:
            return False
        for condition in pod.get("status", {}).get("conditions", []) or []:
            if condition.get("type") == "Ready":
                return condition.get("status") == "True"
        return False

    def _stale_reason(self, pod: Dict[str, Any]) -> Optional[str]:
        """Why the running pod is not the pod this config describes, or None."""
        annotations = (pod.get("metadata") or {}).get("annotations") or {}
        running = annotations.get(_SPEC_HASH)
        wanted = _hash_spec(self.pod_manifest()["spec"])
        if running is None:
            return "it was not created by this device and carries no spec hash"
        if running != wanted:
            return f"spec hash {running} does not match the configured {wanted}"
        return None

    def _describe_tail(self) -> str:
        """Why a pod is not running is in its events, not its status field."""
        done = self._run(["describe", "pod", self.pod], timeout_s=60)
        text = done.stdout.decode("utf-8", "replace")
        marker = text.find("Events:")
        return (text[marker:] if marker != -1 else text)[-1200:].strip() or "(no events)"

    def delete_pod(self, wait: bool = True) -> None:
        """Remove the pod. Used by teardown and by the partial-failure tests."""
        with self._lock:
            self._run(["delete", "pod", self.pod, "--ignore-not-found",
                       "--wait=true" if wait else "--wait=false"], timeout_s=180)
            self._pod_ready = False
            self._synced = False
            self._home = None
            self._cgroup = None

    # ------------------------------------------------------- is the transport honest?

    # Deliberately full of the characters a shell would eat: spaces, both kinds of quote,
    # a dollar sign, parentheses, a brace. If any of them come back changed, something
    # between here and the container is re-parsing our arguments.
    _CANARY = """a b'c"d $HOME (e?f) {g} h"""

    def verify_transport(self) -> None:
        """Prove the kubectl prefix delivers arguments unchanged, before measuring.

        A transport that re-parses arguments does not announce itself. It answers a
        slightly different question and returns a plausible value, which is how a pod
        that was Ready got reported as not Ready and how ``$HOME`` came back as the
        *host's* home directory. One exec at start-up is a cheap price for never having
        to wonder whether that happened.
        """
        with self._lock:
            if getattr(self, "_transport_ok", False):
                return
        expected = self._CANARY
        done = self._exec(["sh", "-c", "printf '%s' " + _sq(expected)], timeout_s=90)
        got = done.stdout.decode("utf-8", "replace")
        # An exec that failed outright is not evidence about quoting, and saying it is
        # sends the reader off to fix a transport that was fine. Two failures, two
        # messages: the pod would not answer, or it answered with the wrong thing.
        if done.returncode != 0 and not got:
            raise DeviceError(
                "could not reach {0}/{1} to check the transport (exit {2}): {3}".format(
                    self.namespace, self.pod, done.returncode,
                    done.stderr.decode("utf-8", "replace")[-400:].strip() or "no output")
            )
        if got != expected:
            raise DeviceError(
                "the kubectl transport for device '{0}' does not pass arguments through "
                "unchanged.\n  sent: {1!r}\n  got : {2!r}\n"
                "A prefix that re-parses its arguments will silently answer the wrong "
                "question rather than fail. If this is WSL, use "
                "['wsl', '-d', '<distro>', '--exec', ...]: without --exec, wsl.exe joins "
                "the arguments and hands them to a login shell.".format(
                    self.device_id, expected, got)
            )
        with self._lock:
            self._transport_ok = True

    # ----------------------------------------------------------------------- paths

    def _container_home(self) -> str:
        if self._home is None:
            done = self._sh('printf %s "$HOME"', timeout_s=60)
            home = done.stdout.decode("utf-8", "replace").strip()
            self._home = home or "/root"
        return self._home

    def resolve(self, path: str) -> str:
        """Expand ``~`` the way the container's shell would.

        The agent invokes binaries without a shell, so an unexpanded ``~`` would reach
        execvp verbatim. Same reasoning as the SSH device, different home directory.
        """
        if path.startswith("~"):
            return posixpath.normpath(self._container_home() + path[1:])
        return path

    def exists(self, path: str) -> bool:
        """Whether the path is in the container, or a loud failure if we could not look.

        The obvious implementation is ``test -e`` and a check of the exit status, and it
        is wrong: ``kubectl exec`` returns non-zero when the pod is momentarily
        unavailable too, so a blip during backend discovery is indistinguishable from a
        missing binary. It then surfaces as "llama-bench not found", is classified as a
        configuration error, and is never retried. Printing a word and insisting on it
        separates "no" from "I could not tell".
        """
        self.ensure_pod()
        target = _sq(self.resolve(path))
        done = self._sh(f"if [ -e {target} ]; then echo MLSYS_YES; else echo MLSYS_NO; fi",
                        timeout_s=60)
        answer = done.stdout.decode("utf-8", "replace").strip()
        if answer.endswith("MLSYS_YES"):
            return True
        if answer.endswith("MLSYS_NO"):
            return False
        raise DeviceError(
            f"could not determine whether {path} exists in {self.namespace}/{self.pod} "
            f"(exit {done.returncode}, pod phase {self.pod_phase()}): "
            + (done.stderr.decode("utf-8", "replace")[-300:] or "no output")
        )

    def first_existing(self, paths: List[str]) -> Optional[str]:
        """Ask about every candidate in one exec instead of one exec each."""
        if not paths:
            return None
        self.ensure_pod()
        done = self._sh(_first_existing_script(paths), timeout_s=90)
        try:
            return _parse_first_existing(done.stdout.decode("utf-8", "replace"))
        except ValueError as exc:
            raise DeviceError(
                f"could not search {self.namespace}/{self.pod} for {len(paths)} "
                f"candidate path(s) (exit {done.returncode}, pod phase "
                f"{self.pod_phase()}): {exc}"
            ) from exc

    # ------------------------------------------------------------------ deployment

    def container_identity(self) -> Optional[Tuple[Any, Any]]:
        """Something that changes whenever the container's filesystem is new.

        The restart count and the current start time together. Either one alone can miss
        a case: a pod recreated from scratch resets the count, and a container that
        restarted keeps its start time only until it does.
        """
        pod = self.pod_json()
        for status in (pod or {}).get("status", {}).get("containerStatuses", []) or []:
            if status.get("name") == self.container:
                running = (status.get("state") or {}).get("running") or {}
                return (status.get("restartCount"), running.get("startedAt"))
        return None

    def sync(self, force: bool = False) -> str:
        """Copy this package into the container. Returns the package root inside it.

        Cheap to call before every run, and it is called before every run on purpose: one
        ``kubectl get pod`` against a benchmark that takes tens of seconds is not a cost
        worth trading for the chance of a sweep dying on a restarted container.
        """
        self.ensure_pod()
        identity = self.container_identity()
        with self._lock:
            if self._synced and not force and identity == self._synced_container:
                return self.workdir
            if self._synced and identity != self._synced_container:
                # Not an error. A restarted container is an empty container, and the
                # only wrong response is to assume otherwise.
                self._transport_ok = False

        self.verify_transport()
        with self._lock:
            package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            blob = _tar_bytes(package_dir, arcname="mlsyslab", skip_pycache=True)

            # Remove first, so a rename in the package does not leave a stale module
            # behind that shadows the new layout.
            prepared = self._sh(
                f"rm -rf '{self.workdir}/mlsyslab' && mkdir -p '{self.workdir}'", timeout_s=120
            )
            if prepared.returncode != 0:
                raise DeviceError(
                    f"could not prepare {self.workdir} in {self.namespace}/{self.pod}: "
                    + prepared.stderr.decode("utf-8", "replace")[-500:]
                )

            done = self._exec(["tar", "-xf", "-", "-C", self.workdir],
                              timeout_s=300, stdin=blob)
            if done.returncode != 0:
                raise DeviceError(
                    f"could not unpack package into {self.namespace}/{self.pod}: "
                    + done.stderr.decode("utf-8", "replace")[-500:]
                )
            self._synced = True
            self._synced_container = identity
            return self.workdir

    def package_root(self) -> str:
        return self.sync()

    @property
    def python_executable(self) -> str:
        return self.python

    # -------------------------------------------------------------------- running

    def _invoke_agent(self, task_json: str, timeout_s: float) -> str:
        root = self.sync()
        argv = ["sh", "-c",
                f"cd '{root}' && PYTHONPATH='{root}' exec {self.python} -m mlsyslab.agent -"]
        done = self._exec(argv, timeout_s=timeout_s, stdin=task_json.encode("utf-8"))
        stdout = done.stdout.decode("utf-8", "replace")

        # A stream that started delivering a result and stopped is the one failure that
        # must never be parsed. Say so by name, and say what the pod was doing.
        if RESULT_BEGIN in stdout and RESULT_END not in stdout:
            raise DeviceError(
                f"kubectl exec stream from {self.namespace}/{self.pod} was truncated: the "
                f"result began but never ended ({len(stdout)} bytes received). Pod phase "
                f"is {self.pod_phase()}. Refusing to parse a partial payload."
            )

        if done.returncode != 0 and not stdout.strip():
            stderr = done.stderr.decode("utf-8", "replace")[-2000:]
            raise DeviceError(
                f"agent in {self.namespace}/{self.pod} failed (exit {done.returncode}), "
                f"pod phase {self.pod_phase()}:\n{stderr}"
            )
        return stdout

    # ------------------------------------------------------------------ file moves

    def push(self, local_path: str, remote_path: str) -> None:
        """Send a file or directory into the container as a tar stream.

        ``remote_path`` names the destination itself, matching the SSH device, so the
        archive is built with the destination's basename and unpacked into its parent.
        """
        self.ensure_pod()
        parent = posixpath.dirname(self.resolve(remote_path)) or "/"
        base = posixpath.basename(self.resolve(remote_path))
        blob = _tar_bytes(local_path, arcname=base, skip_pycache=False)

        made = self._sh(f"mkdir -p '{parent}'", timeout_s=60)
        if made.returncode != 0:
            raise DeviceError("push failed: could not create "
                              + parent + ": "
                              + made.stderr.decode("utf-8", "replace")[-300:])
        done = self._exec(["tar", "-xf", "-", "-C", parent], timeout_s=1800, stdin=blob)
        if done.returncode != 0:
            raise DeviceError("push failed: " + done.stderr.decode("utf-8", "replace")[-500:])

    def pull(self, remote_path: str, local_path: str) -> None:
        self.ensure_pod()
        resolved = self.resolve(remote_path)
        parent = posixpath.dirname(resolved) or "/"
        base = posixpath.basename(resolved)
        done = self._exec(["tar", "-cf", "-", "-C", parent, base], timeout_s=1800)
        if done.returncode != 0 or not done.stdout:
            raise DeviceError("pull failed: " + done.stderr.decode("utf-8", "replace")[-500:])

        destination = os.path.abspath(local_path)
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        staging = destination + ".pull-staging"
        _untar_bytes(done.stdout, staging)
        produced = os.path.join(staging, base)
        if os.path.isdir(produced):
            import shutil

            shutil.copytree(produced, destination, dirs_exist_ok=True)
            shutil.rmtree(staging, ignore_errors=True)
        else:
            os.replace(produced, destination)
            import shutil

            shutil.rmtree(staging, ignore_errors=True)

    # ------------------------------------------------------------- what constrains it

    def cgroup_limits(self) -> Dict[str, Any]:
        """The cgroup ceilings the container is actually running under.

        Read from inside the container rather than taken from the config, because the two
        can disagree: a limit can be dropped by an admission controller, rounded by the
        runtime, or simply never applied. A CPU quota changes throughput, so a record that
        does not carry the *effective* limit is not comparable to one that does.
        """
        if self._cgroup is not None:
            return dict(self._cgroup)

        self.ensure_pod()
        script = (
            "for f in /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max "
            "/sys/fs/cgroup/cpu.weight /sys/fs/cgroup/cpuset.cpus.effective; do "
            'if [ -r "$f" ]; then printf "%s=%s\\n" "$f" "$(cat "$f" 2>/dev/null | tr -d "\\n")"; fi; '
            "done"
        )
        done = self._sh(script, timeout_s=60)
        limits: Dict[str, Any] = {"cgroup_version": 2}
        for line in done.stdout.decode("utf-8", "replace").splitlines():
            key, _, value = line.partition("=")
            name = posixpath.basename(key.strip())
            if name and value:
                limits[name] = value.strip()

        if not any(k.startswith("cpu") or k.startswith("memory") for k in limits):
            # cgroup v1 layout, or a runtime that does not expose the files.
            limits = {"cgroup_version": 1, "note": "cgroup v2 files not readable in container"}

        cpu_max = limits.get("cpu.max")
        if isinstance(cpu_max, str) and cpu_max != "max":
            parts = cpu_max.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                limits["cpu_quota_cores"] = round(int(parts[0]) / int(parts[1]), 4)
        elif cpu_max == "max":
            limits["cpu_quota_cores"] = None

        self._cgroup = limits
        return dict(limits)

    def device_info(self) -> Dict[str, Any]:
        """Everything the base class reports, plus what makes this a pod and not a host."""
        info = super().device_info()
        pod = self.pod_json()
        placement: Dict[str, Any] = {
            "runtime": "kubernetes",
            "namespace": self.namespace,
            "pod": self.pod,
            "container": self.container,
            "image": self.image,
            "image_id": self.image_id(pod),
            "node": self.node_name(pod),
            # Guaranteed, Burstable or BestEffort. It decides the pod's cpu.weight under
            # contention and its eviction order under memory pressure, so two runs in
            # different classes are not interchangeable however similar their limits look.
            "qos_class": (pod or {}).get("status", {}).get("qosClass"),
        }
        try:
            limits = self.cgroup_limits()
        except DeviceError:
            limits = {"error": "unavailable"}
        placement["cgroup"] = limits
        # Hoisted out of the cgroup blob because it is the one number an analysis groups
        # by, and digging it out of a quota pair at read time is how it gets forgotten.
        placement["cpu_quota_cores"] = limits.get("cpu_quota_cores")
        info["placement"] = placement
        return info

    def image_id(self, pod: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """The digest of the image that actually ran, not the tag that was asked for.

        A tag is a moving target. Two runs a week apart against ``:latest`` are not the
        same experiment, and the only thing that proves they were is this string.
        """
        pod = pod if pod is not None else self.pod_json()
        for status in (pod or {}).get("status", {}).get("containerStatuses", []) or []:
            if status.get("name") == self.container:
                return status.get("imageID")
        return None

    def node_name(self, pod: Optional[Dict[str, Any]] = None) -> Optional[str]:
        pod = pod if pod is not None else self.pod_json()
        return (pod or {}).get("spec", {}).get("nodeName")


def _hash_spec(spec: Dict[str, Any]) -> str:
    import hashlib

    blob = json.dumps(spec, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


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


# ---------------------------------------------------------------------- tar helpers

def _tar_bytes(local_path: str, arcname: str, skip_pycache: bool) -> bytes:
    """Archive a local file or directory in memory.

    In memory rather than to a temporary file because the largest thing that travels this
    way is the package itself, a few hundred kilobytes. Model files are mounted into the
    pod from the node, not copied through it.
    """
    if not os.path.exists(local_path):
        raise DeviceError(f"nothing to send: {local_path} does not exist")

    def keep(info: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        if skip_pycache and ("__pycache__" in info.name or info.name.endswith(".pyc")):
            return None
        # Host ownership is meaningless in the container and root-owned files in a
        # non-root image are unreadable, so normalise it away.
        info.uid, info.gid = 0, 0
        info.uname, info.gname = "root", "root"
        return info

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        tar.add(local_path, arcname=arcname, filter=keep)
    return buffer.getvalue()


def _untar_bytes(blob: bytes, destination: str) -> None:
    """Unpack a tar stream, refusing members that would escape the destination."""
    os.makedirs(destination, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r|*") as tar:
        try:
            tar.extractall(destination, filter="data")   # Python 3.12+, and 3.9-3.11 backports
        except TypeError:
            _extract_checked(tar, destination)


def _extract_checked(tar: tarfile.TarFile, destination: str) -> None:
    """extractall for interpreters without the data filter, with the same refusals."""
    root = os.path.abspath(destination)
    for member in tar:
        target = os.path.abspath(os.path.join(root, member.name))
        if not (target == root or target.startswith(root + os.sep)):
            raise DeviceError(f"refusing tar member that escapes the destination: {member.name}")
        if member.issym() or member.islnk():
            raise DeviceError(f"refusing link member in pulled archive: {member.name}")
        tar.extract(member, root)
