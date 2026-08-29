# Curated results

Each directory here is a complete, committed dataset: RunRecords plus the generated
report. Working runs live in `runs/` (gitignored); a dataset moves here when it is
finished, clean, and worth citing.

## containerization/

What containerisation and orchestration cost an inference benchmark, and whether they
change the ranking of the things being compared. Five arms on one i7-12700H: a host
process, and pods with no CPU quota, an 8-core quota, a 4-core quota and a 2-core quota,
all running four threads. 50 records across two independent passes of the matrix, in
`pass1/` and `pass2/`; `REPORT.md` and `containerization.png` are generated from both by
`tools/containerization_report.py`.

The short version: this measurement cannot find a cost of containerisation. Fourteen of
fifteen host-versus-pod comparisons are inside the run-to-run noise, and the mean
difference is under 3% in the pod's favour. A quota *below* the thread count is a
different matter, costing 64% of decode throughput and resolved on every point. The
ranking of the five model/quantisation combinations is identical in all five arms,
including that one, so a comparison run inside a pod picks the same winner as the same
comparison run outside one.

What makes it a measurement rather than an anecdote is in the report's Controls section,
but three things carry most of it: llama-bench is built once and the host arm runs the
binary extracted from the same image, so no difference can be a compiler; the model file
is mounted from the node read-only, so every arm reads the same inode; and the five
"devices" declare a shared resource group, so the arms take turns on the one CPU they
share instead of competing.

Every record carries the cgroup ceilings read from *inside* its own container, so the
quota in the table is the quota that applied rather than the one the config asked for.

## cluster/

Six runs, three genuinely different machines, one YAML, all at once: the Windows laptop
locally, a 2 GB Raspberry Pi 5 over SSH on a tailnet, and a Kubernetes pod with a 4-core
CFS quota on the laptop's Linux side. Small on purpose. It exists to show that the
scheduler and the device abstraction hold across three device kinds at the same time,
not to say anything about the hardware.

`schedule.json` is the interesting file. It records that the laptop and the pod shared a
resource group and took turns while the Pi overlapped both, that two transport failures
were retried and logged, and the clock offsets:

| device | offset | uncertainty | round trip | agent runtime |
|---|---|---|---|---|
| laptop | +0.045 s | 0.061 s | 1.906 s | 1.789 s |
| pi5 | +1.846 s | 2.183 s | 4.374 s | 0.012 s |

Those two rows are worth reading together. Almost all of the laptop's round trip is the
agent's own work collecting sysinfo, and the estimator subtracts it rather than reporting
it as clock error; an earlier version that used only the agent's finishing timestamp
called this machine nearly a second out of step with itself. The Pi's apparent offset is
inside its own uncertainty over the tailnet, so it is reported without a warning: that is
"cannot tell", not "wrong".

The Kubernetes pod had been deleted before this sweep started. The clock check says so,
the first two attempts on it failed and were classified as transport failures rather than
configuration errors, the pod was recreated, and both points completed on the retry. The
run is kept as it happened rather than re-run clean, because that sequence is the thing
worth showing.

## paper12/

The measurements behind "The Memory Wall at the Edge of Language" (submitted to IEEE
Transactions on Computers, 2026), converted into the RunRecord schema by
`tools/backfill_paper12.py`. 29 records: the Pi 5 thread-sweep roofline, the
quantization sweep, PMIC energy per token, and the x86 cross-platform validation up to
7B. The framework reproduces the paper's published fits exactly (Pi 10.7 GB/s, x86
35.7 GB/s, both R^2 = 0.980), which is the correctness check for the whole pipeline.

One record in this set reports 107% of the declared bandwidth ceiling, and the framework
flags it rather than clipping it. The cause is the ceiling, not the measurement. These
records carry the ceiling the paper declared for the laptop, 42.1 GB/s, which came from a
single-stream summation kernel. A two-stream dot-product kernel through BLAS reaches about
54 GB/s on the same machine, and that is the figure every config in this repository now
declares. The paper's number is left in place because changing it would make this
reproduction disagree with the paper it reproduces, which is the opposite of the point.
Read any percentage against a declared ceiling as being only as good as that declaration.

## pi5-campaign/

The first campaign run natively by this framework (August 2026, 43 points, one config:
`configs/pi-overnight.yaml`). Everything paper 12 measured, re-measured in one night
with telemetry the original campaign lacked, plus what it could not measure:

- decode roofline: 10.52 GB/s effective, R^2 = 0.99, 7 models
- TTFT via llama-server streaming: 789 ms at 64 prompt tokens to 12.1 s at 1024 (0.5B)
- per-rail power: DRAM is 2 to 4% of package power during decode
- temperature and throttle state inside every record
- ONNX Runtime vision: int8 is ~11x faster than fp32 on the Cortex-A76
  (resnet18: 1.10 ms vs 12.78 ms), the mirror image of x86 where the same int8 model
  is ~3.6x slower than fp32

Provenance note: five points in the original power block ran while another workload
was on the machine. The per-record `stdev_pct` integrity flag identified all five
(spread 30 to 103% against a campaign norm under 2.5%); they were deleted and
re-measured on the idle machine. The records in this directory are the clean ones.

## laptop-campaign/

The laptop half of the same night (75 points, `configs/laptop-overnight.yaml`), run
behind an automated idle gate after another workload finished:

- size sweep 0.5B to 7B at five thread counts; the 7B reaches 90% of the measured
  bandwidth ceiling, and 20 threads (E-cores engaged) costs the 0.5B 37% of decode
- the canonical FP16-sourced 8-format quantization ladder: Q4_0 is the throughput
  winner (117.6 tok/s), Q6_K and Q8_0 saturate 98% of bandwidth, and the I-quants
  keep pace with K-quants on x86
- context-depth decay: 0.5B decode falls 77 to 32 tok/s from depth 512 to 8192
- TTFT from 367 ms (0.5B, 64-token prompt) to 70.2 s (3B, 4096-token prompt)
- ONNX batch scaling on CPU, and int8 slower than fp32 at every thread count, the
  mirror image of the Pi result

13 records carry a spread flag above 10%: ten are sub-millisecond ONNX inferences
where Windows scheduler jitter is a large fraction of the measurement, three are
20-thread runs where P/E-core scheduling variance is the phenomenon being measured.
They are flagged in-record rather than excluded.

## combined-report/

Tables and figures over everything above at once. Two independent Pi campaigns agree
on the effective bandwidth within 1.7%; the x86 to A76 decode ratio is 3.3 to 3.5x
on every shared model; and the decode roofline holds on both architectures.
