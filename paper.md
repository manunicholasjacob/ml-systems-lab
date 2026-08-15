---
title: 'ml-systems-lab: Reproducible benchmarking of ML inference across heterogeneous hardware'
tags:
  - Python
  - machine learning systems
  - benchmarking
  - reproducibility
  - edge computing
  - energy measurement
authors:
  - name: Manu Nicholas Jacob
    orcid: 0009-0007-6589-6572
    affiliation: 1
affiliations:
  - name: Independent Researcher, Austin, TX, USA
    index: 1
date: 14 August 2026
bibliography: paper.bib
---

# Summary

`ml-systems-lab` is a framework for measuring the performance, energy, and physical state of machine
learning inference across heterogeneous hardware from a single declarative configuration. One YAML
file drives sweeps over models, quantization formats, thread counts, and clock frequencies on a
laptop, a single-board computer reached over SSH, and a GPU, using `llama.cpp` [@llamacpp] and ONNX
Runtime [@onnxruntime] as backends. Each measurement is written to a self-describing JSON record that
carries the hardware, operating system, kernel, compiler, backend version, model, and quantization
alongside the measured metrics (time-to-first-token, prefill and decode throughput, effective memory
bandwidth) and the *physical state of the machine while they were measured*: per-rail power from an
on-board PMIC, temperature, clock frequencies, throttle flags, and CPU utilization. An analysis layer
turns a directory of records into publication-quality tables (plain text, Markdown, and
LaTeX/`booktabs`) and figures, and computes derived artifacts such as memory-bandwidth rooflines
[@williams2009roofline]. The measurement core depends only on the Python standard library so that it
runs unmodified on a 2\,GB Raspberry Pi over SSH, where a scientific Python stack is neither cheap nor
always installable; `numpy` and `matplotlib` are used only for offline analysis and plotting.

# Statement of need

On-device and edge ML inference is increasingly deployed on cheap commodity hardware, yet the numbers
that guide those deployments -- which quantization format, how many threads, which clock, how much
energy per token -- are usually reported without the code, the hardware, the software versions, or the
physical conditions that produced them, and consequently do not reproduce. Existing benchmarking tools
tend to be vendor-specific, target a single backend or accelerator, or record throughput without the
power and thermal context that determines whether a result is trustworthy on a thermally- and
power-constrained device. Standardized efforts such as MLPerf [@mattson2020mlperf] establish rigorous
methodology but are oriented toward submitters with datacenter- or mobile-vendor resources rather than
an individual measuring on an \$80 board.

`ml-systems-lab` fills this gap by treating every number as an artifact with its full physical
provenance, captured on hardware a student or a small team actually owns, and re-runnable from one
config file. It is designed for researchers and practitioners who need cross-device, cross-backend
measurements that are honest about the conditions under which they were taken and reproducible by a
third party. The framework has been used to produce a series of hardware-measurement studies of edge
inference, including a memory-bandwidth roofline for LLM decode, a duty-cycled cold-start
characterization, an energy-optimal configuration analysis, a hybrid-core decode-throughput effect,
and a quantization-format cost study, each released with the archived records the framework emits. By
recording provenance and physical state by construction, it makes such results checkable rather than
merely quotable.

# Key functionality

- **One-config, multi-device sweeps.** A single YAML file specifies models, backends, quantizations,
  thread counts, and clocks; the runner executes them locally, on a remote SBC over SSH, or on a GPU
  host, and collects the results uniformly.
- **Self-describing records.** Every run produces a JSON record capturing hardware, OS, kernel,
  backend build, model, quantization, metrics, and telemetry (power, temperature, clocks, throttle
  flags, utilization), so a record is interpretable without external context.
- **Telemetry.** Optional per-rail power via the Raspberry Pi PMIC, RAPL where available, thermal,
  DVFS, and CPU-utilization sampling, recorded alongside each measurement.
- **Analysis and reporting.** A `report` command aggregates a directory of records into text,
  Markdown, and LaTeX tables and figures, and derives quantities such as decode rooflines and
  thread-scaling curves.
- **A `mlsys` command-line interface** with `probe`, `run`, `report`, `membw`, `compare`, and
  `doctor` subcommands, installable from PyPI (`pip install ml-systems-lab`).

# Acknowledgements

All measurements supported by this framework were performed on the author's own hardware.

# References
