# Changelog

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
