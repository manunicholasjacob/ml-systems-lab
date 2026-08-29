"""A kubectl that answers from a directory instead of a cluster.

This exists so the Kubernetes device is covered by the ordinary test run, on a machine
with no cluster, no container runtime and no network. It is not a Kubernetes emulator: it
implements exactly the calls :class:`~mlsyslab.devices.k8s.K8sDevice` makes, and exits
non-zero on anything else, which makes it a check on the device's own vocabulary as well
as a stand-in for the real thing.

State lives in the directory named by ``FAKE_KUBECTL_STATE``:

  state.json     what the cluster should pretend to be
  fs/            the container's filesystem, as far as anything here is concerned

``state.json`` knobs, all optional:

  ready            bool, whether the pod reports Ready (default true once applied)
  phase            pod phase string (default "Running")
  exists           bool, whether the pod exists at all (default false until applied)
  exec_failure     string, make every exec fail with this on stderr
  truncate_result  bool, cut the agent's output off after the opening sentinel
  mangle_args      bool, behave like a transport that re-parses its arguments
  image_id         imageID to report
  restart_count    how many times the container has restarted
  started_at       when the current container started; change it, with restart_count,
                   to simulate a restart that wiped the container filesystem
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tarfile


def state_dir() -> str:
    return os.environ["FAKE_KUBECTL_STATE"]


def load_state() -> dict:
    path = os.path.join(state_dir(), "state.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    with open(os.path.join(state_dir(), "state.json"), "w", encoding="utf-8") as fh:
        json.dump(state, fh)


def container_fs() -> str:
    path = os.path.join(state_dir(), "fs")
    os.makedirs(path, exist_ok=True)
    return path


def host_path(container_path: str) -> str:
    """Map an absolute container path into the fake container filesystem."""
    return os.path.join(container_fs(), container_path.lstrip("/").replace("/", os.sep))


def fail(message: str, code: int = 1) -> int:
    sys.stderr.write(message + "\n")
    return code


# ------------------------------------------------------------------------ commands

def cmd_get_pod(state: dict) -> int:
    if not state.get("exists"):
        return fail('Error from server (NotFound): pods "mlsyslab" not found')
    ready = "True" if state.get("ready", True) else "False"
    payload = {
        "apiVersion": "v1", "kind": "Pod",
        "metadata": {"name": state.get("pod", "mlsyslab-fake")},
        "spec": {"nodeName": state.get("node", "fake-node")},
        "status": {
            "phase": state.get("phase", "Running"),
            "conditions": [
                {"type": "PodScheduled", "status": "True"},
                {"type": "Ready", "status": ready},
            ],
            "containerStatuses": [
                {"name": "agent",
                 "imageID": state.get("image_id", "sha256:" + "ab" * 32),
                 "restartCount": state.get("restart_count", 0),
                 "state": {"running": {
                     "startedAt": state.get("started_at", "2026-08-28T20:00:00Z")}}},
            ],
        },
    }
    sys.stdout.write(json.dumps(payload))
    return 0


def cmd_apply(state: dict) -> int:
    manifest = sys.stdin.buffer.read()
    with open(os.path.join(state_dir(), "applied.json"), "wb") as fh:
        fh.write(manifest)
    state["exists"] = True
    state.setdefault("ready", True)
    save_state(state)
    sys.stdout.write("pod/mlsyslab-fake created\n")
    return 0


def cmd_delete(state: dict) -> int:
    # Only existence. Whether a *future* pod becomes ready is a property of the pretend
    # cluster that the test configured, not something a delete gets to overwrite.
    state["exists"] = False
    save_state(state)
    return 0


def run_agent(script: str) -> int:
    """Run the real agent, the way the container would, on this machine."""
    parts = shlex.split(script)
    # cd '<root>' && PYTHONPATH='<root>' exec python3 -m mlsyslab.agent -
    root = parts[1]
    payload = sys.stdin.buffer.read()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env["FAKE_CONTAINER_ROOT"] = root
    done = subprocess.run([sys.executable, "-m", "mlsyslab.agent", "-"],
                          input=payload, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env=env)
    output = done.stdout
    if load_state().get("truncate_result"):
        from mlsyslab.agent import RESULT_BEGIN

        marker = output.find(RESULT_BEGIN.encode())
        if marker != -1:
            output = output[: marker + len(RESULT_BEGIN) + 40]
    sys.stdout.buffer.write(output)
    sys.stderr.buffer.write(done.stderr)
    return done.returncode


def run_shell(script: str, state: dict) -> int:
    """The handful of shell constructs the device actually emits."""
    if state.get("mangle_args"):
        # A transport that re-parses: quotes are gone and $HOME belongs to the host.
        script = script.replace("'", "").replace("$HOME", "/home/host-not-container")

    if "mlsyslab.agent" in script:
        return run_agent(script)

    if script.startswith("printf '%s' "):
        sys.stdout.write(shlex.split(script[len("printf '%s' "):])[0])
        return 0

    if script.startswith('printf %s "$HOME"'):
        sys.stdout.write("/root")
        return 0

    if state.get("mangle_args") and script.startswith("printf %s "):
        # What the canary looks like after a transport has eaten its quotes. It still
        # succeeds and still prints something, which is exactly why the check exists:
        # the real failure answered a different question rather than failing. Ordered
        # after the $HOME case on purpose, because this pattern would swallow it.
        sys.stdout.write(script[len("printf %s "):])
        return 0

    if script.startswith("for p in ") and "MLSYS_HIT" in script:
        # for p in 'a' 'b' 'c'; do ...
        listing = script[len("for p in "):].split("; do", 1)[0]
        for candidate in shlex.split(listing):
            if os.path.exists(host_path(candidate)):
                sys.stdout.write("MLSYS_HIT:" + candidate)
                return 0
        sys.stdout.write("MLSYS_NONE")
        return 0

    if script.startswith("if [ -e "):
        target = shlex.split(script[len("if [ -e "):].split("]")[0])[0]
        present = os.path.exists(host_path(target))
        sys.stdout.write("MLSYS_YES" if present else "MLSYS_NO")
        return 0

    if script.startswith("for f in /sys/fs/cgroup"):
        for line in state.get("cgroup_lines", ["cpu.max=200000 100000",
                                               "memory.max=4294967296"]):
            sys.stdout.write(line + "\n")
        return 0

    made = False
    for chunk in script.split("&&"):
        chunk = chunk.strip()
        if chunk.startswith("rm -rf "):
            import shutil

            shutil.rmtree(host_path(shlex.split(chunk[len("rm -rf "):])[0]),
                          ignore_errors=True)
            made = True
        elif chunk.startswith("mkdir -p "):
            os.makedirs(host_path(shlex.split(chunk[len("mkdir -p "):])[0]), exist_ok=True)
            made = True
    if made:
        return 0

    return fail(f"fake kubectl does not implement this shell script: {script!r}", 127)


def cmd_exec(argv: list, state: dict) -> int:
    if state.get("exec_failure"):
        return fail(state["exec_failure"], 1)
    if not state.get("exists"):
        return fail("Error from server (NotFound): pods \"mlsyslab\" not found")

    if argv[:2] == ["sh", "-c"]:
        return run_shell(argv[2], state)

    if argv and argv[0] == "tar":
        return cmd_tar(argv)

    return fail(f"fake kubectl does not implement exec of {argv!r}", 127)


def cmd_tar(argv: list) -> int:
    if "-xf" in argv:
        destination = host_path(argv[argv.index("-C") + 1])
        os.makedirs(destination, exist_ok=True)
        blob = sys.stdin.buffer.read()
        import io as _io

        with tarfile.open(fileobj=_io.BytesIO(blob), mode="r|*") as tar:
            tar.extractall(destination)
        return 0
    if "-cf" in argv:
        source = host_path(argv[argv.index("-C") + 1])
        name = argv[-1]
        import io as _io

        buffer = _io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            tar.add(os.path.join(source, name), arcname=name)
        sys.stdout.buffer.write(buffer.getvalue())
        return 0
    return fail(f"fake kubectl does not implement tar {argv!r}", 127)


def main(argv: list) -> int:
    state = load_state()

    # Strip the flags the device may add before the verb.
    index = 0
    while index < len(argv):
        if argv[index] in ("-n", "--namespace", "--kubeconfig", "--context"):
            index += 2
        else:
            break
    argv = argv[index:]
    if not argv:
        return fail("no command")

    verb = argv[0]
    if verb == "get" and argv[1:2] == ["pod"]:
        return cmd_get_pod(state)
    if verb == "create" and argv[1:2] == ["namespace"]:
        return 0
    if verb == "apply":
        return cmd_apply(state)
    if verb == "delete":
        return cmd_delete(state)
    if verb == "describe":
        sys.stdout.write("Events:\n  fake event\n")
        return 0
    if verb == "exec":
        rest = argv[1:]
        if rest and rest[0] == "-i":
            rest = rest[1:]
        # <pod> -c <container> -- <argv...>
        separator = rest.index("--")
        return cmd_exec(rest[separator + 1:], state)
    return fail(f"fake kubectl does not implement {verb!r}", 127)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
