# Changelog

## 0.2.0 (2026-08-28)

Kubernetes, concurrent scheduling with partial failure, and observability.

The measured result this release exists for: on one i7-12700H, running the same binary in
a Kubernetes pod rather than as a host process changes decode throughput by under 3% on
average, and 14 of 15 host-versus-pod comparisons cannot be resolved from the run-to-run
noise at all. A CPU quota *below* the benchmark's thread count costs 64%, resolved on
every point. The ranking of five model and quantisation combinations is identical in all
five arms, so a comparison run inside a pod picks the same winner as one run outside it.
50 records over two independent passes, in `results/containerization/`, with the method,
controls and limitations in its REPORT.md.

### Added

- **`K8sDevice`**: a third implementation of the existing `Device` interface, so a
  container in a pod is a device in the same sense a laptop and a Pi over SSH are. No
  backend changed, and a test enforces that no backend so much as mentions Kubernetes.
  - `kubectl` is an argv prefix rather than a binary path, so a cluster reached through
    `wsl -d Ubuntu --exec k3s kubectl`, a jump host, or a plain `kubectl` all work.
  - `verify_transport()` refuses to measure anything through a transport that re-parses
    its arguments, because such a transport does not fail, it answers a different
    question.
  - Files move as a tar stream over `exec`, which is what `kubectl cp` does minus the
    host-path translation that breaks when the harness and kubectl see different
    filesystems.
  - A truncated `kubectl exec` stream is reported as truncated, never parsed.
  - Pods carry an `mlsyslab/spec-hash` annotation; a pod whose spec no longer matches the
    config is replaced rather than reused under yesterday's limits.
  - The package sync is keyed to the container's lifetime, not the session's: a restarted
    container is an empty container.
- **Concurrent scheduling** (`mlsyslab.scheduler`, `mlsys run --concurrent`), alongside
  the sequential runner rather than replacing it.
  - Per-device concurrency limits, defaulting to 1.
  - `resource_group`, for devices that are secretly the same machine, with round-robin
    turn-taking between them.
  - Circuit breaker per device: closed, open, half-open probe, closed.
  - Bounded retries with jittered exponential backoff, and only for transport failures.
  - `retries.jsonl` and `schedule.json`, plus a warning on any record that was retried.
  - Points on a device that never came back are written as failed records that say they
    were never measured, so the hole stays visible and a resumed sweep picks it up.
  - Opt-in `failover_to`, with the moved run tagged and given its own run id.
  - A clock-skew check before every concurrent sweep, using Cristian's algorithm.
- **Prometheus exporter and Grafana dashboard** (`mlsyslab.prometheus`, `mlsys export`,
  `mlsys run --prometheus-port`, `grafana/mlsyslab-sweep.json`). Standard library only.
  The records stay the source of truth; the exporter is never in the write path.
- **`docker/Dockerfile`**: the agent image, base pinned by digest, llama.cpp pinned by
  commit. The harness is streamed in at run time rather than baked in.
- **`k8s/`**: namespace and pod manifests, generated from the device by
  `tools/write_manifests.py` and drift-checked by a test. `k8s/cloud/` holds the cloud GPU
  node's manifests, runbook and teardown script.
- **`configs/containerization.yaml`**: the five-arm study this release is built around.
- **`configs/cluster.yaml`** and `results/cluster/`: one sweep across a Windows laptop, a
  2 GB Raspberry Pi 5 over SSH and a Kubernetes pod, running at once, from one file.
- **`tools/containerization_report.py`**: its analysis, which refuses to call a
  difference real unless it clears the measurement's own noise.
- `DeviceInfo.placement`: where a run executed and what constrained it, including the
  effective cgroup ceilings read from inside the container. A run under a CPU quota is
  not comparable to one without, and the record now carries its own quota.
- `Device.first_existing()`: backend discovery in one round trip rather than one per
  candidate path.
- Per-device `env` for the llama.cpp backend, for the machine-specific parts of an
  environment such as where a build's shared objects live.
- The agent reports `agent_started_epoch_s` as well as `agent_epoch_s`.
- Analysis columns: `cpu_quota_cores`, `runtime`, `qos_class`, `node`, `image_id`,
  `repetitions`, `warmup`.
- `docs/DISTRIBUTED.md`, `docker/README.md`, `k8s/README.md`, `k8s/cloud/RUNBOOK.md`.

### Fixed

- `exists()` returned False when it could not run the check at all, so a momentary
  outage during backend discovery surfaced as "llama-bench not found", was classified as
  a configuration error, and was therefore never retried. It now raises when it cannot
  tell. Fixed in both the SSH and Kubernetes devices.
- A `DeviceError` raised while building a task was classified as a configuration error
  rather than a transport failure, so a device that dropped during discovery was written
  off instead of retried.
- `Runner` is now safe to drive from several threads: device and backend construction and
  the append to `index.jsonl` are guarded. The sequential path is unchanged.
- `SSHDevice.sync()` could delete the package tree out from under a running agent when
  called concurrently.
- The clock-skew estimate used only the agent's finishing timestamp, so the agent's own
  runtime was reported as clock error: four devices sharing one physical clock came out
  two to three seconds apart.

### Changed

- Version 0.1.2 was a PyPI-only republish of 0.1.1 and has no separate entry here.

## 0.1.1 (2026-08-13)

Reproducibility and onboarding release.

### Added

- `mlsys doctor` command: checks Python version, dependencies, llama.cpp binaries,
  and (with `--config`) validates model paths and device reachability. Run this first
  when something is not working.
- `python -m mlsyslab` support (`__main__.py`), so the package works without the
  console script installed.
- `configs/example-smoke.yaml`: a ready-to-edit template config with download
  instructions and placeholder paths. Copy, fill in your paths, and run.
- `runs/thread-cliff/`: 33-point thread-scaling experiment (P-core/E-core decode
  cliff on Alder Lake i7-12700H).

### Fixed

- README test count corrected to 70 (was 65).
- Quick start now leads with `mlsys doctor` and the template config before showing
  the full multi-device example.

## 0.1.0 (2026-08-10)

First release. Everything below was built and validated against real hardware in one
campaign cycle: an i7-12700H laptop (Windows), a 2 GB Raspberry Pi 5 over SSH, and an
RTX 3050 through ONNX Runtime DirectML.

### Framework

- One YAML config expands to a full experiment matrix (`mlsys run`), resumable via
  deterministic run ids; failures are records, not lost exceptions.
- Device abstraction: local machine and SSH devices behind one interface, with a
  stdlib-only agent pushed to the device so benchmark and telemetry sampler are
  colocated. Nothing is installed on the device under test.
- Backends: llama.cpp (`llama-bench` for throughput, `llama-server` streaming for
  time-to-first-token and end-to-end latency) and ONNX Runtime (latency percentiles,
  batch throughput, arbitrary execution providers).
- Telemetry: Raspberry Pi PMIC per-rail power (core vs DRAM), Intel RAPL, sysfs
  thermal zones and throttle bits, per-core CPU utilization, DVFS control. The
  two-run rule (throughput and power never from the same sampled run) is encoded.
- Analysis: dataset filtering/grouping, roofline fits, tables in text/Markdown/LaTeX,
  publication figures, one-command `REPORT.md` (`mlsys report --full`).
- Tools: `mlsys probe` (capability map), `mlsys membw` (measured DRAM ceiling with a
  stability check), `mlsys compare` (two result sets side by side).

### Data

- `results/paper12/`: backfilled measurements behind an IEEE Transactions on
  Computers submission; the framework reproduces the paper's fits exactly.
- `results/pi5-campaign/`: 43-point native campaign (roofline 10.52 GB/s, R^2 0.99;
  TTFT; per-rail power; ONNX vision with the 11x int8 win on Cortex-A76).
- `results/laptop-campaign/`: 75-point campaign (0.5B-7B size sweep to 90% of
  bandwidth ceiling, 8-format quant ladder, context-depth decay, TTFT grid, ONNX
  batch scaling, and int8 losing to fp32 on x86).

### Quality

- 70 hardware-free tests (recorded fixtures); CI on 3 OS x 3 Python versions plus a
  no-dependencies job proving the measurement core runs bare.
- Per-record integrity flags (run-to-run spread, throttle state, ceiling violations),
  validated by a real contamination incident that the flags caught.
