# Raspberry Pi 5 power data index

Several artifact repositories hold power measurements taken on the same kind of board, a
Raspberry Pi 5 (Cortex-A76, 2 GB), through the same on-board instrument. Each one stores
them in its own layout. This page says where the power data is in each repository, what
a file holds, the units, how often the instrument was read, and how to cite it.

It describes files only. It adds no new measurements, and the results themselves are in
each repository's README and manuscript.

| repository | what holds power data | granularity | nominal read period | licence | concept DOI (latest version) |
|---|---|---|---|---|---|
| [edge-llm-memory-wall](https://github.com/manunicholasjacob/edge-llm-memory-wall) | `results/L3_energy.json`, `results/L5_dvfs.json` | mean power per configuration | 100 ms | MIT | [10.5281/zenodo.21844855](https://doi.org/10.5281/zenodo.21844855) (v1.0.2, 10.5281/zenodo.23329221) |
| [edge-breakeven-speedup](https://github.com/manunicholasjacob/edge-breakeven-speedup) | 6 files in `data/`, 27 of 34 files in `data/v2/` | one record per measured window, with sample count and span | 23 ms (v2) | MIT | [10.5281/zenodo.21987261](https://doi.org/10.5281/zenodo.21987261) (v2.0.0, 10.5281/zenodo.23341310) |
| [edge-cold-start-tax](https://github.com/manunicholasjacob/edge-cold-start-tax) | `results/e4_energy.json`, `results/e6_dvfs.json`, `results/e8_ondevice.json` | per model and phase, per clock policy, per request stream | 12 ms (E4), 20 ms (E6, E8) | MIT | [10.5281/zenodo.21844857](https://doi.org/10.5281/zenodo.21844857) (v1.0.2, 10.5281/zenodo.23329220) |
| [barrier-spin-crossover](https://github.com/manunicholasjacob/barrier-spin-crossover) | 8 `*.power.jsonl` in `data/pi-cortex-a76/energy-and-tightened/` | **raw per-sample traces**, every rail | 100 ms | MIT | [10.5281/zenodo.22163331](https://doi.org/10.5281/zenodo.22163331) (v1.0.0, 10.5281/zenodo.22163332) |
| ml-systems-lab (this repository) | `telemetry` block of Pi 5 RunRecords in `results/pi5-campaign/` (and a backfill of L3 above in `results/paper12/pi5-2gb/`) | mean power per run | 100 ms (10 Hz default) | MIT | [10.5281/zenodo.21867055](https://doi.org/10.5281/zenodo.21867055) (v0.2.2, 10.5281/zenodo.23329333) |

Only barrier-spin-crossover ships the individual samples. Everywhere else the samples were
reduced to a mean or an integral on the device, and the stored record is that reduction.

The data in every repository above is unchanged between its latest Zenodo version and the
default branch as of 14 October 2026, so the DOI and the GitHub files are the same data.

## The instrument, common to all of them

- **Source.** `vcgencmd pmic_read_adc`, which prints a voltage and, for most rails, a
  current for each rail the board's power-management IC monitors.
- **Total power** is the sum of V x I over every rail that reports both a voltage and a
  current. A rail that reports only a voltage is not summed.
- **Core power** is the `VDD_CORE` rail. **DDR power** is the sum of the rails whose names
  start with `DDR` (`DDR_VDD2`, `DDR_VDDQ`). On this board `DDR_VDDQ` never populates its
  current channel, so the DDR figure is carried by `DDR_VDD2` alone (stated in the
  barrier-spin-crossover README, and visible in its traces).
- **The rails** present in the barrier-spin-crossover traces are `0V8_AON`, `0V8_SW`,
  `1V1_SYS`, `1V8_SYS`, `3V3_ADC`, `3V3_DAC`, `3V3_SYS`, `3V7_WL_SW`, `DDR_VDD2`, `DDR_VDDQ`,
  `HDMI` and `VDD_CORE`.
- **What it is not.** The rails sit downstream of the board's converters, so the total is
  not wall power and not 5 V input power. It covers the wireless, HDMI and IO rails as
  well as the SoC, so it is not comparable to a package-power number such as RAPL.
- **It runs on the machine under test.** Each read spawns a process. The nominal period is
  a `sleep` between reads, so the achieved rate is lower than the nominal one whenever a
  read takes time. Where a repository records the sample count and span, the achieved
  rate is listed below; where it does not, only the nominal period is known.
- **Timestamps** are Python `time.perf_counter()` (memory-wall, cold-start, breakeven) or
  `time.monotonic()` (barrier-spin). Neither is wall-clock time.
- **Low-clock limit.** In the edge-breakeven-speedup control `E_purecore2` (N processes on
  a floating-point loop with no inference workload, at 1500 and 2400 MHz), the summed rails
  at 1500 MHz did not track the work done as more cores became busy, and that repository's
  manuscript withdraws every power quantity measured at 1500 MHz. Any power value in this
  index taken at 1500 MHz with more than one busy core should be treated the same way
  unless you validate the instrument yourself. The sections below say where each
  file records its clock, so these rows can be found.

## edge-llm-memory-wall

llama.cpp decode on the Pi 5. Power is in two files; the rest of `results/` is timing,
memory and perplexity.

| file | contents | units |
|---|---|---|
| `results/L3_energy.json` | `_meta.P_idle_W`, then per model: `decode_ts`, `P_decode_W`, `P_core_W`, `P_ddr_W`, `ddr_share_pct`, `energy_per_tok_J`, `energy_per_tok_mJ`, `dyn_energy_per_tok_mJ`, `file_mb` | W, J, mJ, tokens/s, MB |
| `results/L5_dvfs.json` | per model, per clock (keys in kHz): `decode_ts`, `P_W`, `energy_per_tok_mJ`, `edp` | W, mJ, tokens/s |

How it was taken (`scripts/llm_common.py`, `decode_energy`): throughput comes from a
`llama-bench` run with no sampler, and power from a second, separately sampled run of the
same command at a 100 ms period. Power is the mean over that run after its first 4 s (load
and prefill). Energy per token is that mean power divided by the unsampled decode rate.
Idle power is the mean over a 4 s quiet window at the same period. The clock is pinned at
2.4 GHz for L3. Sample counts are not stored.

## edge-breakeven-speedup

ONNX Runtime CNN inference across thread counts and clocks. Two campaigns: `data/` (first,
superseded by the repository's own account) and `data/v2/` (current). Every file is JSON
Lines, one object per measured window.

**Power fields**, the same in every file that has them, from `code/v2/pmic.py`
(`PmicSampler.integrate`, trapezoidal over the window):

| field | units |
|---|---|
| `n_pmic_samples`, `pmic_span_s` | count, s |
| `energy_total_J`, `energy_core_J`, `energy_ddr_J` | J |
| `P_mean_W`, `P_core_mean_W`, `P_ddr_mean_W` | W (energy divided by span) |
| `ddr_share_pct` | percent of total energy |

Files also carry a per-task quantity such as `energy_per_inf_mJ`, `mJ_per_inf`,
`mJ_per_batch`, `energy_per_gemm_mJ` or `energy_per_token_mJ`, plus clock
(`freq_khz`, `freq_actual_khz`, kHz) and temperature (`temp_before_c`, `temp_after_c`, C).

**Achieved rate.** The v2 sampler's period is set to 23 ms. Computed from the stored
`(n_pmic_samples - 1) / pmic_span_s`, the per-file median is 20.8 to 21.7 Hz in every v2
file that has power fields, except `E_samplerate`, which varies the period on purpose (23,
50, 100 and 200 ms). The first campaign's files give 21.0 to 21.4 Hz by the same
computation; its sampler module is not shipped.

**Files with power fields:**

- `data/`: `P15_model_energy`, `P15b_batch_threshold`, `P15c_resolution`,
  `P15d_resolution_more`, `P15e_1500mhz_recheck`, `P2_exit_energy` (831 records in all).
- `data/v2/`: `E_bimodal`, `E_ci`, `E_cotenant`, `E_cotenant2`, `E_duty`, `E_duty2`,
  `E_duty3`, `E_floordrift`, `E_gemm`, `E_governor`, `E_idle`, `E_idleE`, `E_idleci`,
  `E_idlegov`, `E_isolate`, `E_llm`, `E_purecore2`, `E_recheck15`, `E_samplerate`,
  `E_shuffle`, `E_spin`, `E_spinE`, `E_spinidle`, `E_sustained`, `E_sustained3`,
  `E_threads`, `E_threads3`.
- `data/v2/` files **without** power: `E_clockv` (firmware clock and core voltage),
  `E_ops`, `E_ops3` (per-operator profiles), `E_perf`, `E_perf2`, `E_perfN`, `E_perfspin`
  (Arm PMU counters).

Instrument checks inside the dataset: `E_samplerate` (same cell, four sampler periods),
`E_floordrift` (forty back-to-back 30 s idle windows at 2.4 GHz), `E_isolate` (sampler
pinned off the workload's cores) and `E_purecore2` (described above).

`E_threads3` records carry `stage: "threads"`, the same tag as `E_threads`; tell them
apart by file name.

## edge-cold-start-tax

ONNX Runtime cold wake, warm wake and steady inference for twelve models. Power is in three
result files; `e5_residency.json` holds energy numbers too, but they come from a simulation
driven by the E4 constants, not from the instrument.

| file | contents | units |
|---|---|---|
| `results/e4_energy.json` | `_meta.P_idle_W`, `_meta.P_idle_core_W`; per model: `P_cold_W`, `P_warm_W`, `P_steady_W`, `E_wake_cold_J`, `E_wake_warm_J`, `E_inf_steady_J`, the matching durations (`t_wake_cold_s`, `t_wake_warm_s`, `lat_steady_s`), operation counts `n_cold`, `n_warm`, `n_steady`, and two derived ratios | W, J, s |
| `results/e6_dvfs.json` | per model, per clock policy (`schedutil`, `pinned_min`, `pinned_max`): `wake_ms`, `wake_E_J` | ms, J |
| `results/e8_ondevice.json` | per workload, per memory budget, per policy: `E_total_J`, `dur_s`, `E_per_req_J`, latency and miss counts | J, s, ms |

How it was taken (`scripts/cold_common.py`, `scripts/e4_energy.py`):

- **E4** pools every sample that falls inside any window of one phase across many repeated
  operations, takes their mean as that phase's power, and multiplies it by the phase's
  median duration. The repository chose this because a single sub-100 ms wake holds too
  few samples to integrate. Nominal period 12 ms. Clock pinned at 2.4 GHz.
- **E6** integrates each wake trapezoidally at a 20 ms nominal period and stores the median
  of six trials. A wake whose window held fewer than two samples yields no energy, so five
  of the fifteen policy cells that carry `wake_E_J` are `null`; `vit-base` has no
  `wake_E_J` at all. `pinned_min` is 1500 MHz (see the low-clock limit above).
- **E8** integrates over a whole request stream at a 20 ms nominal period, clock pinned
  at 2.4 GHz.

Sample counts are not stored, so the achieved rate is unknown.

## barrier-spin-crossover

llama.cpp decode under different OpenMP spin settings. The only repository here with raw
traces. All power data is in `data/pi-cortex-a76/energy-and-tightened/`; the laptop data
in `data/laptop-alderlake/` has no power.

| file | contents |
|---|---|
| `G{1,2}_*_sc{N}.power.jsonl` | one JSON object per sample: `t` (`time.monotonic()`, s), `total_w` (W, sum of rails), then one key per rail in W |
| `G{1,2}_*_sc{N}.sampled.json` | the `llama-bench -o json` output of the sampled run |
| `G{1,2}_*_sc{N}_clean.json` | the `llama-bench -o json` output of the matching unsampled run (use this for throughput) |
| `H_hog2_sc{N}_r{1,2}.json` | the denser contended sweep, `llama-bench -o json` output only, no power |
| `windows.jsonl` | per tag: `spincount`, `bench_start`, `bench_end` (same clock as `t`), `rc` |

`G1` is idle and `G2` has two CPU-bound competing processes; `N` is `GOMP_SPINCOUNT`
(0, 1e4, 1e5, 1e6). The eight traces hold 183 to 963 samples each, at an achieved 10.0 Hz
(median interval), with no error rows. Trim a trace to `[bench_start, bench_end]` from
`windows.jsonl` before averaging. The README records the governor as `performance` and the
ARM clock at 2400 MHz on every Pi run. The sampler is `scripts/pmic_sampler.py`, which also
writes `{"t", "error"}` rows on a failed read; none occur in the shipped traces.

## ml-systems-lab

Pi 5 RunRecords whose `telemetry` block carries power: 13 of the 43 records in
`results/pi5-campaign/pi5/` and 5 of the 22 in `results/paper12/pi5-2gb/`. Fields are
`power_w_mean`, `power_w_core`, `power_w_dram` (W), and, depending on the record,
`energy_j` (J), `energy_per_token_mj` (mJ), `idle_power_w` (W), `sample_hz` and
`sample_count`. The reader is `src/mlsyslab/telemetry/power.py` (`read_pmic`), with the
same rail grouping as above, at 10 Hz by default. The pi5-campaign records also carry
`freq_khz_mean`, `freq_khz_min` and `freq_khz_max`. The five paper12 records are not a
separate measurement: they are edge-llm-memory-wall's `results/L3_energy.json` converted
into the RunRecord schema by `tools/backfill_paper12.py` (tagged `backfill`, with
`raw.source` naming the file), so they were taken at 2.4 GHz like L3. The measurement rules are in
[Measurement method](METHOD.md). `results/README.md` records that five points of the
original pi5-campaign power block ran beside another workload and were deleted and
re-measured; the shipped records are the re-measured ones.
