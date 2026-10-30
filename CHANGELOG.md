# Changelog

## Unreleased

## 0.2.3 (2026-10-30)

Every result under `results/` keeps its records. One measurement change, on the
accelerator path only: `membw --device` now sizes its buffers so all three fit the
free-memory budget, which shrinks the default working set on cards with less than about
4 GiB free. One metadata change: on Windows without `nvidia-smi`, the `gpu` field of new
records names the discrete card rather than the first adapter listed.

### Added

- **`membw` reports whether the ceiling was measured on a busy machine.** New
  `telemetry/contention.py` measures the share of the machine's CPU capacity used by other
  processes over the measurement window, and `membw.measure` returns it as
  `foreign_cpu_fraction`. Above 10% the result also carries `contended`,
  `busiest_other_processes` and a `warning`. `telemetry.capabilities()` gains
  `cpu_contention`.

  This covers the case the `membw` docstring already named and the stability flag cannot
  see: steady background load depresses every repetition equally, so the scatter stays
  small while the ceiling is wrong. Two measured instances now. A package manager cost this
  lab a factor of 3.6. On a Galaxy M21 running Android's background dexopt job, the same
  kernels with the same arguments read 6.35 GB/s against 9.32 GB/s on the same quiet
  device, 32% low, at stability 0.85 and 0.87. That job runs at nice 10, which yields the
  CPU but not memory bandwidth, and Android schedules it when the device is idle and
  charging, which is the state a person creates in order to benchmark a phone.

  Works on Linux, Android and Windows, because it divides by the aggregate CPU counter,
  which already sums over every logical processor, so the platform's units cancel and no
  clock rate or core count is needed. Process names are Linux and Android only; Windows
  still reports the fraction. `None` means the check could not run, never that the machine
  was idle. Observed: idle Pi 5 0.017, Windows laptop in ordinary use 0.205 to 0.228,
  Android with dexopt around 0.6.

- **Documentation site** at https://manunicholasjacob.com/ml-systems-lab/, built from
  `docs/` with MkDocs and deployed from `main` (`.github/workflows/docs.yml`, a
  `--strict` build on every pull request that touches it). `pip install -e ".[docs]"`.
- **Example config gallery**: six copyable templates in `configs/examples/`
  (quantization ladder, thread scaling, TTFT latency, a board over SSH, ONNX Runtime on
  the CPU, two machines at once) and a generated page, `docs/configs.md`, that lists
  every config with its devices, backend, mode and run count from the real loader.
  `tests/test_config_gallery.py` loads every shipped config and fails if the page is
  stale.
- **Agent image workflow** (`.github/workflows/publish-image.yml`): a portable
  `GGML_NATIVE=OFF` build for amd64 and arm64 on native runners, smoke-tested, pushed
  to `ghcr.io/manunicholasjacob/mlsyslab-agent` on a release; pull requests that touch
  `docker/` build and test it without pushing.
- Feature-request and pull-request templates, issue-chooser links to Discussions and the
  docs, and `SECURITY.md` (private vulnerability reporting).
- **Raspberry Pi 5 power data index** (`docs/DATA_INDEX.md`): which files in this and four
  artifact repositories hold PMIC power records, their fields, units, sampling period,
  licence and DOI. No new measurements.

### Security

- **The `governor:` config value is quoted before it reaches a privileged shell**
  (`telemetry/dvfs.py`). `_sudo_write` runs `sudo -n sh -c` on a script it builds, and the
  governor name arrived from the config file unquoted, so a config carrying
  `ondemand; touch /tmp/x` ran the second half as root on a machine set up with the
  passwordless sudo this module documents. `shlex.quote` closes it, and it adds nothing to
  an ordinary governor name, so the script for a real sweep is unchanged.
  `tests/test_dvfs.py` is new and one of its cases runs the generated script against a
  fake sysfs tree to show the hostile value is written rather than executed.

  Raised by 0x73746F66 while reviewing `GCVE-110-PYPI-2026-017962`, which classified
  versions 0.2.1 and 0.2.2 as malware. That record was **withdrawn as a false positive on
  24 October 2026**, with an OpenVEX `not_affected` statement; the two lines it matched were
  a README example and an sdist-only test. This hardening is the one real finding the
  review produced, and no released version was ever malicious.

### Fixed

- **The governor is put back the way it was found, and put back without being asked**
  (`telemetry/dvfs.py`, `agent.py`). Two defects, found because a Raspberry Pi 5 was left on
  the `performance` governor after a run and the next measurement inherited it. `restore()`
  defaulted to `schedutil` while its docstring promised to put the machine back the way it was
  found; Raspberry Pi OS runs `ondemand`, so on that board the default moved the machine to a
  governor it had never been on. `set_governor` now records what it finds the first time it is
  called and `restore()` with no argument puts that back, returning `False` rather than guessing
  when there is nothing recorded. Separately, the agent restored only when the config also set
  `restore_governor`, so a config that set `governor:` and omitted it left the machine changed;
  it now restores whenever it changed the governor, and `restore_governor` still overrides the
  target for a caller who wants a specific one.

- **`membw --device` first run on a real card, and the defect it found** (issue #1). On an
  RTX 3050 Laptop GPU the ceiling is 160.1 to 161.2 GB/s across three runs, 91% of the
  176.03 GB/s its own memory clock implies, with stability above 0.98. The free-memory
  budget covered one buffer while three are resident, so on that 4 GiB card the default
  used 93% of free memory and `--mb 1300` oversubscribed it; under WDDM that paged
  silently and cut `copy_GBs` by a fifth. The budget now covers all three buffers, and
  the record carries `resident_mb`. The run is kept in `tests/recorded/` and the
  summarising step (`membw.summarise_device`) is tested against it.

- **Per-core CPU utilisation on Windows** (issue #2). `cpu_util_pct_per_core` was absent
  on every Windows record because `GetSystemTimes` is system-wide only. Each logical
  processor's idle, kernel and user times now come from
  `NtQuerySystemInformation(SystemProcessorPerformanceInformation)` through ctypes, with
  no new runtime dependency. The list is reported only when, pooled, it gives the
  `GetSystemTimes` busy fraction to within 2 points over the same interval (on any
  platform), and it is absent on machines with more than 64 logical processors, where that
  call sees one processor group. On a 20-thread laptop the pooled and aggregate figures
  agree to under half a point over 0.5 s, a core pinned busy reads 100.0, and the
  per-core figures track psutil's to a mean of 1.2 to 1.6 points, psutil reading higher
  because it adds interrupt and DPC time that kernel time already contains. A recorded
  snapshot pair from that laptop is in `tests/recorded/`; psutil joins the `dev` extra
  for the live cross-check.

- **The GPU name on Windows without `nvidia-smi` prefers a discrete card.** The fallback
  took the first adapter Windows listed, which on many laptops is the integrated GPU, so
  a record could name the Intel or AMD iGPU while the run used the discrete card. A
  discrete name (NVIDIA, Radeon RX/Pro, Arc A/B-series) now wins, then any real adapter,
  then the Basic Display and virtual-monitor drivers; ties keep the listed order. Records
  where `nvidia-smi` answered are unaffected.

### Changed

- **Manuscript wording that cannot go stale.** The `.zenodo.json` description and two
  README lines said a manuscript was "under review at the IEEE Internet of Things
  Journal". A Zenodo record is permanent, so a review state frozen into it stops being
  true at the first decision. Both now give the dated fact instead: submitted to that
  journal in September 2026.
- CI also tests Python 3.10 and 3.11 on Linux, so every version the package metadata
  claims is tested somewhere; 3.11 is what the agent runs under on Raspberry Pi OS and
  in the agent image.
- `results/README.md` describes the ONNX vision rows as measurements of those
  CIFAR-sized files rather than as a speedup factor, and notes that the Pi's int8
  files are static QDQ exports.
- Package metadata gives the licence as the SPDX expression `MIT` with `license-files`
  (PEP 639) instead of the deprecated table and classifier, which setuptools warned about
  on every build; the build now needs setuptools 77 or later.
- README: a Related tools section linking the sibling measurement tools, one line each.

## 0.2.2 (2026-10-12)

No measurement, record format or analysis changes; every result under `results/` is
untouched.

### Changed

- **The Prometheus exporter's default port is now 10058**, the next free number on the
  Prometheus default-port allocations page. 9109 belongs to the Graphite exporter. `mlsys export`
  defaults to 10058, and the README, configs, Grafana note and runbooks use it. If you
  scrape a running sweep, change the target or pass `--prometheus-port 9109` /
  `--port 9109` to keep the old one.
- The README quickstart runs straight after `pip install`: `mlsys probe` and
  `mlsys membw` need no config. The full YAML-driven run is shown from a clone, because
  `configs/` is not part of the package.
- PyPI page: keywords, classifiers and Documentation and DOI links; pipx and uv install
  lines.
- CI installs the built wheel outside the checkout and runs the README's first command
  block from an empty directory (`tools/verify_quickstart.py`), so a quickstart that only
  works from a clone fails in CI.

### Fixed

- On Windows 11 24H2 and later, which no longer ship `wmic`, records said
  "Intel64 Family 6 Model ..." for the CPU and carried no physical core count or GPU name.
  The CPU name now comes from the registry, and the core count and GPU name from CIM.

## 0.2.1 (2026-10-03)

A patch release. No measurement, record format or analysis changes, and every result
under `results/` is untouched.

### Fixed

- `mlsys membw --reps 0` printed a ceiling of 0.00 GB/s, with no kernel and no thread
  count, and offered it as the device's `dram_peak_GBs`. `--reps` and `--mb` on
  `mlsys membw` and on `python -m mlsyslab.membw`, and `--limit`, `--max-attempts` and
  `--max-workers` on `mlsys run`, must now be at least 1; anything else is a usage error
  with exit code 2. `membw.measure()` and `membw.measure_device()` refuse `reps < 1`
  themselves.
- `python -m mlsyslab.membw --device` documented seven repetitions on an accelerator but
  always ran five, because `--reps` defaulted to 5. It now runs 5 on the host and 7 on
  the device unless told otherwise.
- A config file with a syntax error raised a bare `JSONDecodeError` or `YAMLError` that
  gave a line and column but not the file. It is now a `ConfigError` naming the file.
- The paper-12 data was described as a submission to IEEE Transactions on Computers in the
  README, `results/README.md`, the backfill docstring and the Zenodo description. That
  manuscript has since been retitled and is under review at the IEEE Internet of Things
  Journal, so each place now names that venue. "Published fits" is now "reported fits", since none
  of the manuscripts is published.
- The README said the test suite was 113 tests taking about 20 seconds. It is now
  described without a count that goes stale with every release.

### Changed

- Python 3.14 is tested in CI on Linux, macOS and Windows and declared in the package
  classifiers.
- Removed `du.exe.stackdump`, an MSYS2 crash dump committed by accident with 0.1.2, and
  ignore `*.stackdump`.

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

- Version 0.1.2 was tagged in git (`v0.1.2`) but never reached PyPI, which goes
  0.1.1 then 0.2.0. It has no separate entry here. An earlier note in this file
  described it as a PyPI-only republish of 0.1.1, which is the wrong way round.

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
  TTFT; per-rail power; ONNX vision on CIFAR-sized inputs).
- `results/laptop-campaign/`: 75-point campaign (0.5B-7B size sweep to 90% of
  bandwidth ceiling, 8-format quant ladder, context-depth decay, TTFT grid, ONNX
  batch scaling).

### Quality

- 70 hardware-free tests (recorded fixtures); CI on 3 OS x 3 Python versions plus a
  no-dependencies job proving the measurement core runs bare.
- Per-record integrity flags (run-to-run spread, throttle state, ceiling violations),
  validated by a real contamination incident that the flags caught.
