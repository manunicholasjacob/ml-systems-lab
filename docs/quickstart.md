# Quick start

## 1. Install, and look at the machine

Python 3.9 or newer, on Linux, macOS or Windows.

```
pip install ml-systems-lab                 # or: pipx install ml-systems-lab / uv tool install ml-systems-lab
mlsys probe                                # what this machine can measure
mlsys membw                                # its DRAM read ceiling, and the config line to record it
```

`mlsys membw` prints a `dram_peak_GBs` line. Keep it: every utilization percentage the
framework reports is a fraction of that number, so it is measured, never looked up.

Nothing here builds llama.cpp or ONNX Runtime for you. Point a config at a build you
already have (llama.cpp's `llama-bench`, plus `llama-server` for TTFT), and
`mlsys doctor` will tell you what it cannot find.

## 2. Read a dataset before producing one

The repository ships complete datasets, so the analysis side can be tried with no
hardware at all:

```
git clone https://github.com/manunicholasjacob/ml-systems-lab && cd ml-systems-lab
mlsys report results/pi5-campaign                # 43 measured points from a Raspberry Pi 5
mlsys report results/combined-report --full      # tables and figures across both machines
```

## 3. One machine

Copy a template from [the gallery](configs.md), fill in the lines marked `# <-- EDIT`,
and run it. The smallest is `configs/example-smoke.yaml`:

```
mlsys doctor --config configs/example-smoke.yaml   # check everything is wired up
mlsys run configs/example-smoke.yaml --dry-run     # see the matrix, run nothing
mlsys run configs/example-smoke.yaml               # run it
mlsys report runs/smoke-test                       # tables to the terminal
mlsys report runs/smoke-test --full                # REPORT.md plus PNG/PDF figures
mlsys report runs/smoke-test --format latex        # booktabs, ready to paste
```

Run campaigns on an idle machine. The framework records the run-to-run spread of every
point and flags any above 10%, but no number of repetitions rescues a measurement from
constant background load. [Measurement method](METHOD.md) has the rest of the rules.

## 4. A board over SSH

A board is a device block with a host, a user and a key. The agent is copied to it as
source and executed with the board's own Python, so nothing is installed there.
`configs/examples/sbc-over-ssh.yaml` is the template:

```
mlsys membw --device sbc --config configs/examples/sbc-over-ssh.yaml
mlsys doctor --config configs/examples/sbc-over-ssh.yaml
mlsys run configs/examples/sbc-over-ssh.yaml
```

On a Raspberry Pi 5 the records also carry per-rail PMIC power (core versus DRAM),
temperature and the firmware throttle bits. Measure power in its own block, as the
template does: the sampler perturbs decode, so throughput is taken from runs without it.

## 5. Several machines at once

Add `--concurrent` and the matrix is scheduled across every device in it, subject to
each device's `max_concurrency`:

```
mlsys run configs/examples/two-machines.yaml --concurrent --prometheus-port 10058
```

Only transport failures are retried, every retry is written down, and a device that
stops answering trips a circuit breaker rather than stalling the sweep. Two device ids
that are really one machine (a host and a pod on it) declare a shared `resource_group`
and take turns. [Several machines at once](DISTRIBUTED.md) covers the design;
[Agent image and k3s](container.md) covers running the agent in a Kubernetes pod.

## 6. Compare, export, share

```
mlsys compare runs/before runs/after --metric decode_tps   # same workloads, side by side
mlsys export runs/my-sweep                                 # Prometheus exposition format
```

Measured something on hardware that is not in the shipped datasets? Open a
[results-gallery issue](https://github.com/manunicholasjacob/ml-systems-lab/issues/new/choose)
with the output of `mlsys probe` and `mlsys report`. Records from new machines are the
most useful contribution the project can receive.
