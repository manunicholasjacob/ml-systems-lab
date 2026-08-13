# Measurement methodology

Every rule in this document was paid for with a wrong number. They are encoded in the
framework where possible; where a rule needs the operator's cooperation, the framework
detects violations and flags them in the record instead of failing.

## The machine must be idle

Memory-bandwidth-bound workloads are exquisitely sensitive to background load, and the
failure mode is silent: the run completes, the number is plausible, and it is wrong.
The same laptop measured 50.7 GB/s idle and 14.1 GB/s while a package manager finished
in the background, a 3.6x error with nothing crashed.

Best-of-N repetitions cannot save you from *constant* background load, only from
transient spikes. The framework therefore records run-to-run spread (`stdev_pct`) in
every record, and the analysis layer flags any dataset where spread exceeds 10%.

Rule of thumb: campaign runs happen overnight or on a machine you are not touching.

## Throughput and power are measured in separate runs

Reading the Raspberry Pi PMIC spawns a process per sample. At 50 Hz the sampler alone
slowed decode by roughly 6x; the measurement destroyed the measurand. The framework
samples at 10 Hz by default, and the recommended pattern (encoded in the configs) is:

1. a throughput run with `sample_power: false`, whose tokens/s you trust,
2. a power run with `sample_power: true`, whose watts you trust.

The two are joinable because the workload is deterministic. What you must not do is
report tokens/s and watts from the same sampled run.

RAPL on x86 is nearly free to read and does not need this split, but the configs keep
the same shape on both platforms so records stay comparable.

## Thermal state is part of the result

A run that thermally throttles is not a failed run; it is a measurement of a different
operating point. Temperature (start, mean, max), clock (mean, min, max) and the Pi's
live throttle bits are recorded inside every record, not in a side file, so a throttled
run cannot be silently averaged with a cool one. Sweeps should use `prepare.settle_s`
to let the board return to baseline between points; the record's `temp_c_start` shows
whether that actually happened.

## Tool selection for llama.cpp

* **`llama-bench`** is the only measurement path used for throughput. It is
  non-interactive, does its own warmup and repetitions, and reports prefill and decode
  separately in JSON.
* **`llama-server` streaming** is the only path for TTFT and end-to-end latency.
  A throughput benchmark structurally cannot tell you when the first token arrived.
  Requests are issued with `temperature 0`, fixed seed, and `cache_prompt: false`,
  because a warm prompt cache would turn the second repetition into a different
  experiment. The reported TTFT is the median across repetitions.
* **`llama-cli` is never used.** It blocks on interactive input even with interactive
  mode disabled; a scripted invocation once hung for 77 minutes.

Known llama-bench sharp edges the backend absorbs:

* there is no context-size flag; context depth is swept with `-d` (`-c` is rejected),
* `-p 0` produces an anomalously low decode rate, so a real prompt (>= 32 tokens) is
  always supplied when decode is being measured.

## Memory hygiene on small boards

On a 2 GB board the margin between "fits" and "swaps" is thinner than one model. Three
rules, all automated in `prepare`:

* **Kill nothing by pattern, ever.** A `pkill -f` aimed at a stale process once killed
  a running campaign repeatedly. The framework launches once and kills by PID.
* **Reset swap and drop caches between models** (`reset_swap`, `drop_caches`). Leftover
  resident pages from a previous model force the next one to stream from SD, which
  looks like a 13x slowdown and reports as a successful run.
* **Record memory state before and after every run.** When a number looks wrong, the
  first question is "was swap in use", and the record already answers it.

The build-time corollary: compiling llama.cpp on the board needs `-j1`; parallel builds
OOM and take the machine into swap-death.

## Bandwidth ceilings are declared, not probed

Roofline utilization needs a denominator. The framework will not invent one: a device
config either declares `dram_peak_GBs` (your measured read ceiling, or the datasheet
figure, your choice as long as you say which) or utilization is simply not reported.

Any measured ceiling is a *lower bound* on what the memory system can do. When decode
bandwidth exceeds the declared ceiling, the record carries a warning saying the ceiling
is too low; it does not clamp, and it does not print >100% silently.

## Failures are data

A model that does not fit, a server that will not start, a device that dropped off the
network: each produces a record with `status: failed`, the error, and the raw output.
Resume logic retries failures (they are usually environmental) but never re-runs
successes. Holes in a matrix are visible in the results directory rather than being
discovered during analysis.

## Reproducing a run

```
git clone <repo> && cd ml-systems-lab
pip install -e ".[dev]"
python -m pytest -q          # 70 tests, no hardware needed
mlsys probe                  # what this machine is and can measure
mlsys run configs/smoke.yaml # ~1 minute wiring check
```

For a remote device: SSH key auth (`BatchMode` must succeed), Python 3.9+, and the
inference backend built on the device. `mlsys probe --config <your config>` verifies
all of it without running a benchmark. Nothing is installed on the device; the agent is
copied to `~/.mlsyslab` and imported from there.

To reproduce the paper-12 tables and figures from the shipped records:

```
mlsys report results/paper12 --full
```
