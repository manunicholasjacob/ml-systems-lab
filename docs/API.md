# Using ml-systems-lab from Python

The command line covers running a sweep and turning the results into tables. This page is
for the other cases: reading records someone else produced, analysing them your own way,
and adding a backend or a device kind.

Every example here runs against the datasets shipped in `results/`, so you can paste any
of them into a shell after `pip install -e .` and see output without owning a Pi.

## What is stable, and what is not

| Area | Stability |
|---|---|
| `RunRecord` and its nested dataclasses, and the JSON they serialise to | Stable within a `schema_version`. Fields are added, never repurposed, and `from_dict` drops fields it does not know, so old records stay loadable. A change to what an existing field means bumps `SCHEMA_VERSION`. |
| `mlsyslab.schema.load_records`, `from_dict`, `compute_run_id` | Stable. |
| `Dataset` and its query methods, `decode_roofline`, `linear_fit_through_origin` | Stable in signature; new keys may appear in returned dicts. |
| The `Backend` and `Device` contracts | Stable. Implementing them is the supported way to extend the framework. |
| `analysis.tables.*` and `analysis.plots.FigureSet` | The rendering is presentation, so exact strings and figure layout may change between minor versions. |
| Everything with a leading underscore | Private. |

## Reading records

A run record is one JSON file. Nothing needs the framework to read it, which is the
point, but the loader gives you dataclasses instead of nested dicts.

```python
from mlsyslab.schema import load_records

records = load_records("results/pi5-campaign")
print(len(records), "records")

r = records[0]
print(r.run_id, r.status)
print(r.device.cpu, r.device.machine, r.device.ram_bytes)
print(r.backend.name, r.backend.version, r.backend.build)
print(r.workload.model, r.workload.quantization, r.knobs.threads)
print(r.metrics.decode_tps, "tok/s at", r.telemetry.temp_c_mean, "C")
```

`load_records` takes a file, a directory, or a directory tree, and skips anything that is
not a record. Failed runs load like any other, so filter when you mean to:

```python
ok = [r for r in records if r.ok]
failed = [r for r in records if not r.ok]
for r in failed:
    print(r.run_id, r.error)
    print(r.raw.get("stdout", "")[:400])   # what the backend actually printed
```

That last line is the reason failures are records rather than exceptions. The raw output
is kept, so a parser fixed months later can be re-run against a campaign that has long
since finished.

### The record, in one place

* `device`: `cpu`, `machine` (the ISA), core counts, `ram_bytes`, `os`, `kernel`,
  `board_model`, `capabilities`, and `dram_peak_GBs` as declared in the config. That last
  one is declared, never probed, because a wrong denominator turns silently into a wrong
  utilization percentage.
* `backend`: `name`, `version`, `build` (llama.cpp build number or commit), the resolved
  `binary` path, and the `execution_provider` where one applies.
* `workload`: `model`, `quantization`, `model_bytes` and `weight_bytes`, prompt and output
  token counts, `context_tokens`, `batch_size`.
* `knobs`: `threads`, `governor`, `freq_khz`, `repetitions`, and anything else the
  operator set for this run.
* `metrics`: TTFT, prefill and decode throughput, latency percentiles, peak RSS, derived
  decode bandwidth and roofline utilization, run-to-run spread. Every field is optional.
  A backend that cannot measure something leaves it `None`, never zero.
* `telemetry`: per-rail power, energy, temperature at start, mean and max, throttle flags,
  CPU utilization, clock frequency, and the sampling rate that produced them.
* `raw`: the backend's own output.

`r.to_dict()` and `r.to_json()` round-trip through `from_dict`.

## Analysing a directory

`Dataset` wraps a list of records with the filtering and grouping that the reports use.

```python
from mlsyslab.analysis.dataset import Dataset, decode_roofline

ds = Dataset.from_directory("results/pi5-campaign")   # failed runs excluded by default
print(len(ds), "records")
print(ds.unique("device"), ds.unique("model"))

for (device,), subset in ds.filter(backend="llamacpp").group_by("device"):
    fit = decode_roofline(subset)
    print(device, round(fit["bw_eff_GBs"], 2), "GB/s effective,",
          "R^2", round(fit["r2"], 4), "over", fit["n"], "models,",
          round(fit["peak_fraction_pct"], 1), "% of the declared DRAM ceiling")
```

`decode_roofline` fits one device's worth of records, so group first. It takes the best
decode rate per model across thread counts rather than every point, because the roofline
is a claim about the ceiling and folding sub-saturated thread points into it smears a
thread effect into a size effect.

Three more query methods carry most of the weight:

```python
ds.having("energy_per_token_mj")      # only runs that actually measured the thing
ds.sort("decode_tps", reverse=True)
ds.best("decode_tps")                 # one row, or None
for warning in ds.integrity_warnings():   # spread, throttling, missing denominators
    print(warning)
```

`having` exists because most analyses are only meaningful over the subset that measured
the quantity in question, and treating a missing measurement as zero is the failure this
guards against.

The fit is `decode_tps = BW_eff / model_bytes` through the origin. It is the same fit the
papers publish, which is how the framework was checked: run the loop above over
`results/paper12/` and it returns 10.7 GB/s for the Pi 5 and 35.7 GB/s for the laptop,
both at R^2 = 0.980, matching the submitted manuscript.

Rows as plain dicts, for pandas or anything else:

```python
from mlsyslab.analysis.dataset import flatten

rows = [flatten(r) for r in ds.records]
# import pandas as pd; df = pd.DataFrame(rows)
```

Tables render to three formats from the same call:

```python
from mlsyslab.analysis import tables

print(tables.summary_table(ds))                    # aligned text
open("table.md", "w").write(tables.roofline_table(ds, fmt="markdown"))
open("table.tex", "w").write(tables.energy_table(ds, fmt="latex"))
```

Figures need matplotlib and are grouped in `FigureSet`:

```python
from mlsyslab.analysis.plots import FigureSet

figs = FigureSet(ds, "figures")          # PNG and PDF by default
figs.roofline()
figs.thread_scaling()
print(figs.written)

FigureSet(ds, "figures").generate_all()  # every figure the data supports
```

A figure whose data is absent returns `None` rather than drawing an empty axis, so
`generate_all` on a dataset without power measurements simply produces no energy figure.

## Running a sweep from Python

The CLI is a thin wrapper over these three steps, so doing it yourself is reasonable when
the sweep is generated rather than written.

```python
from mlsyslab.config import load
from mlsyslab.runner import Runner, console_reporter

config = load("configs/example-smoke.yaml")
print(len(config.specs), "runs planned")

runner = Runner(config, output_dir="runs/from-python", on_event=console_reporter())
records = runner.run()                   # or runner.run(config.specs[:5]) for a subset
```

A `Config` can also be built in memory: `load` accepts JSON as well as YAML, which is what
the no-dependency CI job uses to prove the measurement path works without PyYAML.

Two behaviours worth knowing before you call `run`:

* **Resume is the default.** Each spec hashes to a deterministic `run_id` with timestamps
  excluded, so a campaign that was interrupted picks up where it stopped, and deleting one
  record makes exactly that point re-execute. Pass `resume=False` to force everything.
* **A failure does not stop the campaign.** It is written as a record with `status` set to
  `failed`, and the run continues.

## Adding a backend

Three methods against the contract in `backends/base.py`.

```python
from mlsyslab.backends.base import Backend, RunSpec
from mlsyslab.schema import BackendInfo, RunRecord

class MyBackend(Backend):
    name = "mybackend"

    def discover(self, device) -> BackendInfo:
        """Find the binary or library on the device and report its version."""

    def build_task(self, spec: RunSpec, device) -> dict:
        """One spec in, one agent task out. The task is JSON: argv, env, timeouts."""

    def parse(self, spec: RunSpec, result: dict, device) -> RunRecord:
        """The agent's raw output in, one RunRecord out. Parsing happens here, on the
        host, never on the device."""
```

Register it in `backends/__init__.py`, lazily if it imports anything heavy. Write the
parser test from a recorded output before you point it at hardware; every backend in the
repository has one, and that is why the suite needs no devices.

## Adding a device kind

Subclass `Device` from `devices/base.py` and implement `_invoke_agent`, `push`, `pull`,
`resolve`, `exists` and `package_root`. The base class handles the agent protocol,
capability probing and binary discovery.

The one hard constraint: the agent payload has to reach the device as source and run under
its own Python with no third-party imports. That is what keeps a 2 GB single-board computer
a first-class device rather than a special case.

`LocalDevice` runs the agent as a subprocess rather than in-process, which looks wasteful
and is not: `getrusage(RUSAGE_CHILDREN)` accumulates across children, so an in-process
agent would corrupt peak RSS on every run after the first.

## The agent protocol

Useful if you are debugging a device rather than extending the framework. The agent is a
single stdlib-only module that takes one JSON task on stdin and prints one JSON result
between sentinels:

```
python -m mlsyslab.agent - <<< '{"kind": "command", "argv": ["echo", "hi"], "sample_power": false}'
```

```
===MLSYSLAB-RESULT-BEGIN===
{"status": "ok", "stdout": "hi\n", "rusage": {...}, "telemetry": {...}}
===MLSYSLAB-RESULT-END===
```

The sentinels exist because a login shell on a remote box prints banners, motd text and
sometimes warnings around anything you run. Everything outside them is discarded.
