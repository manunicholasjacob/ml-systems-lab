"""The payload that executes on the device under test.

This module is the reason the framework can treat a laptop and a Raspberry Pi over SSH as
the same thing. It runs *on* the device, so the benchmark process and the telemetry
sampler are local to each other. The alternative, sampling from the host over SSH, would
put a network round trip inside every sample and make power and temperature useless.

Contract: a task dict arrives on stdin as JSON, a result dict leaves on stdout as JSON
between two sentinels. Sentinels rather than bare stdout because SSH cheerfully mixes in
motd banners, sudo lecture text and warnings that would otherwise corrupt the parse.

Standard library only, deliberately. It has to run on a 2 GB Pi we do not want to install
a scientific stack onto, and the host does all parsing and analysis anyway.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from . import sysinfo
from .telemetry import TelemetrySession, capabilities, dvfs, power

RESULT_BEGIN = "===MLSYSLAB-RESULT-BEGIN==="
RESULT_END = "===MLSYSLAB-RESULT-END==="


# --------------------------------------------------------------------- environment

def prepare_environment(options: Dict[str, Any]) -> List[str]:
    """Put the machine into a known state. Returns warnings, never raises.

    The steps here are all scar tissue from earlier campaigns on the 2 GB Pi:

    * Leftover processes from a killed run keep RAM occupied, the model gets swapped out,
      and decode runs about 13x slower because weights stream from the SD card. The run
      looks finished and the number is garbage.
    * Swap that is already dirty from a previous model has the same effect, so it is
      cycled off and on rather than trusted.
    * Page cache holds the previous model, which makes a cold-start measurement warm.
    """
    warnings: List[str] = []

    if options.get("reset_swap"):
        ok = _sudo("swapoff -a && swapon -a")
        if not ok:
            warnings.append("swap reset failed; a previously swapped model may still be resident")

    if options.get("drop_caches"):
        ok = _sudo("sync && echo 3 > /proc/sys/vm/drop_caches")
        if not ok:
            warnings.append("drop_caches failed; page cache may make a cold run look warm")

    governor = options.get("governor")
    if governor and not dvfs.set_governor(governor):
        warnings.append(f"could not set governor to {governor}; ran at the default instead")

    freq = options.get("freq_khz")
    if freq and not dvfs.set_frequency_khz(int(freq)):
        warnings.append(f"could not pin clock to {freq} kHz; ran at the default instead")

    settle = float(options.get("settle_s") or 0)
    if settle > 0:
        # Let the board cool and the clock ramp down between points, so a sweep does not
        # measure the thermal history of the previous point.
        time.sleep(settle)

    return warnings


def _sudo(script: str) -> bool:
    try:
        done = subprocess.run(
            ["sudo", "-n", "sh", "-c", script], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def memory_state() -> Dict[str, Any]:
    """Free memory and swap in use, recorded because both explain a slow run."""
    state: Dict[str, Any] = {}
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in ("MemAvailable", "MemFree", "SwapFree", "SwapTotal"):
                    state[key] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return {}
    if "SwapTotal" in state and "SwapFree" in state:
        state["SwapUsed"] = state["SwapTotal"] - state["SwapFree"]
    return state


# ---------------------------------------------------------------------- task: run

def task_command(task: Dict[str, Any]) -> Dict[str, Any]:
    """Run one command while sampling, and return its raw output plus telemetry.

    The agent does not parse the command's output. Parsing lives on the host so that a
    parser bug found next month can be fixed by re-reading stored stdout instead of by
    re-running an eight hour campaign.
    """
    argv = task["argv"]
    timeout = float(task.get("timeout_s") or 3600)
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (task.get("env") or {}).items()})

    session = TelemetrySession(
        sample_hz=float(task.get("sample_hz") or 10.0),
        sample_power=bool(task.get("sample_power", True)),
    ).start()

    mem_before = memory_state()
    t0 = time.perf_counter()
    timed_out = False
    try:
        done = subprocess.run(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, timeout=timeout, env=env,
            cwd=task.get("cwd") or None,
        )
        returncode = done.returncode
        stdout = done.stdout.decode("utf-8", "replace")
        stderr = done.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = -1
        stdout = (exc.stdout or b"").decode("utf-8", "replace")
        stderr = (exc.stderr or b"").decode("utf-8", "replace")
    except OSError as exc:
        session.stop()
        return {"status": "failed", "error": f"could not execute {argv[0]}: {exc}"}
    t1 = time.perf_counter()
    session.stop()

    # Skip the leading warmup window when summarising, so model load does not get
    # averaged into the inference power figure.
    skip = float(task.get("skip_leading_s") or 0)
    telemetry = session.summary(t0 + skip, t1, tokens=task.get("tokens_for_energy"))

    return {
        "status": "ok" if returncode == 0 and not timed_out else "failed",
        "error": ("timed out after %.0f s" % timeout) if timed_out else (
            None if returncode == 0 else f"exit code {returncode}"
        ),
        "duration_s": t1 - t0,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr[-20000:],   # tails are what matter; heads are banner noise
        "telemetry": telemetry,
        "memory_before": mem_before,
        "memory_after": memory_state(),
    }


# ------------------------------------------------------- task: streaming latency

def task_server_latency(task: Dict[str, Any]) -> Dict[str, Any]:
    """Measure true time-to-first-token against an OpenAI-style streaming endpoint.

    This exists because throughput benchmarks cannot answer the question users actually
    feel. ``llama-bench`` reports aggregate prefill and decode rates over a batch; it
    never issues a single request, so it cannot say when the first token appeared.
    ``llama-cli`` can, but it blocks on interactive input and hangs when scripted, so the
    server's streaming API is the only reliable path.
    """
    import urllib.error
    import urllib.request

    server_argv = task["server_argv"]
    host = task.get("host", "127.0.0.1")
    port = int(task.get("port", 8080))
    base = f"http://{host}:{port}"
    startup_timeout = float(task.get("startup_timeout_s") or 300)
    requests = task.get("requests") or []

    proc = None
    warnings: List[str] = []
    try:
        proc = subprocess.Popen(
            server_argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
    except OSError as exc:
        return {"status": "failed", "error": f"could not start server: {exc}"}

    load_start = time.perf_counter()
    ready = False
    while time.perf_counter() - load_start < startup_timeout:
        if proc.poll() is not None:
            output = (proc.stdout.read() or b"").decode("utf-8", "replace") if proc.stdout else ""
            return {"status": "failed",
                    "error": f"server exited during startup with code {proc.returncode}",
                    "stderr": output[-8000:]}
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.5)
    load_ms = (time.perf_counter() - load_start) * 1000.0

    if not ready:
        _terminate(proc)
        return {"status": "failed", "error": f"server not ready within {startup_timeout:.0f} s"}

    session = TelemetrySession(
        sample_hz=float(task.get("sample_hz") or 10.0),
        sample_power=bool(task.get("sample_power", True)),
    ).start()

    results: List[Dict[str, Any]] = []
    window_start = time.perf_counter()
    for spec in requests:
        payload = {
            "prompt": spec.get("prompt", "Hello"),
            "n_predict": int(spec.get("n_predict", 128)),
            "temperature": 0.0,     # deterministic, so repeats are comparable
            "seed": 0,
            "stream": True,
            "cache_prompt": False,  # otherwise the second request measures a cache hit
        }
        results.append(_one_streaming_request(base, payload, spec))
    window_end = time.perf_counter()

    session.stop()
    _terminate(proc)

    good = [r for r in results if r.get("ttft_ms") is not None]
    total_tokens = sum(r.get("output_tokens") or 0 for r in good)
    telemetry = session.summary(window_start, window_end,
                                tokens=total_tokens if total_tokens else None)

    if not good:
        return {"status": "failed", "error": "every streaming request failed",
                "requests": results, "telemetry": telemetry}

    return {
        "status": "ok",
        "load_ms": load_ms,
        "duration_s": window_end - window_start,
        "requests": results,
        "telemetry": telemetry,
        "warnings": warnings,
    }


def _one_streaming_request(base: str, payload: Dict[str, Any],
                           spec: Dict[str, Any]) -> Dict[str, Any]:
    """Issue one streaming completion, timing the first token and the last."""
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base + "/completion", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )

    t_send = time.perf_counter()
    t_first: Optional[float] = None
    tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=float(spec.get("timeout_s", 900))) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if not body or body == "[DONE]":
                    continue
                try:
                    chunk = json.loads(body)
                except ValueError:
                    continue
                if chunk.get("content"):
                    if t_first is None:
                        t_first = time.perf_counter()
                    tokens += 1
                if chunk.get("stop"):
                    break
    except Exception as exc:
        return {"error": str(exc), "prompt_tokens": spec.get("prompt_tokens")}
    t_end = time.perf_counter()

    if t_first is None:
        return {"error": "stream produced no content", "prompt_tokens": spec.get("prompt_tokens")}

    decode_s = t_end - t_first
    return {
        "prompt_tokens": spec.get("prompt_tokens"),
        "output_tokens": tokens,
        "ttft_ms": (t_first - t_send) * 1000.0,
        "e2e_latency_ms": (t_end - t_send) * 1000.0,
        # Tokens after the first, over the time after the first: the steady-state rate,
        # uncontaminated by prefill.
        "decode_tps": (tokens - 1) / decode_s if tokens > 1 and decode_s > 0 else None,
    }


def _terminate(proc) -> None:
    """Stop the server without leaving an orphan holding a model in RAM.

    An orphaned server on a 2 GB board is not a tidiness problem. It keeps the weights
    resident, the next run swaps, and that run reports a number roughly an order of
    magnitude too low while looking entirely successful.
    """
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=15)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=10)
        except Exception:
            pass


# -------------------------------------------------------------------------- main

TASKS = {
    "command": task_command,
    "server_latency": task_server_latency,
    "sysinfo": lambda task: {"status": "ok"},
}


def execute(task: Dict[str, Any]) -> Dict[str, Any]:
    kind = task.get("kind", "command")
    handler = TASKS.get(kind)
    if handler is None:
        # Fall through to the decoration below: even a bad task deserves an
        # attributable result, so the host can tell which device rejected it.
        warnings: List[str] = []
        result: Dict[str, Any] = {"status": "failed",
                                  "error": f"unknown task kind: {kind}"}
    else:
        warnings = prepare_environment(task.get("prepare") or {})
        try:
            result = handler(task)
        except Exception as exc:  # a crashed agent must still report, not hang the campaign
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            restore = (task.get("prepare") or {}).get("restore_governor")
            if restore:
                dvfs.restore(restore)

    result.setdefault("warnings", [])
    result["warnings"] = list(result["warnings"]) + warnings
    result["device"] = sysinfo.collect(task.get("device_id"), task.get("device_kind", "local"))
    result["capabilities"] = capabilities()
    result["agent_time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if task.get("measure_idle_power"):
        idle = power.measure_idle_power_w()
        if idle is not None:
            result.setdefault("telemetry", {})["idle_power_w"] = idle
    return result


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("-",):
        with open(argv[0], "r", encoding="utf-8") as fh:
            task = json.load(fh)
    else:
        task = json.loads(sys.stdin.read() or "{}")

    result = execute(task)
    sys.stdout.write(RESULT_BEGIN + "\n")
    sys.stdout.write(json.dumps(result, default=str))
    sys.stdout.write("\n" + RESULT_END + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
