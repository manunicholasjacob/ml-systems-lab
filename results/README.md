# Curated results

Each directory here is a complete, committed dataset: RunRecords plus the generated
report. Working runs live in `runs/` (gitignored); a dataset moves here when it is
finished, clean, and worth citing.

## paper12/

The measurements behind "The Memory Wall at the Edge of Language" (submitted to IEEE
Transactions on Computers, 2026), converted into the RunRecord schema by
`tools/backfill_paper12.py`. 29 records: the Pi 5 thread-sweep roofline, the
quantization sweep, PMIC energy per token, and the x86 cross-platform validation up to
7B. The framework reproduces the paper's published fits exactly (Pi 10.7 GB/s, x86
35.7 GB/s, both R^2 = 0.980), which is the correctness check for the whole pipeline.

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
