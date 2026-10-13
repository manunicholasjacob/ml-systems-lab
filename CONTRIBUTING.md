# Contributing

Three kinds of contribution help most, roughly in this order:

1. **Results from hardware the project has not seen.** Open a results-gallery issue
   with the output of `mlsys probe` and `mlsys report <dir>`. A single new machine,
   from a single-board computer to a server, is worth more than a feature.
2. **A config that answers a question people ask.** See "Adding an example config"
   below.
3. **Code**: a backend, a device kind, a telemetry source, or a fix. The rest of this
   file is about that.

Questions go to [Discussions](https://github.com/manunicholasjacob/ml-systems-lab/discussions);
bugs and numbers that look wrong go to issues.

## Running the tests

```
pip install -e ".[dev]"
python -m pytest -q
```

The suite is hardware-free: recorded llama-bench output and canned agent results stand
in for real devices, so it runs identically on a laptop and in CI. If a change needs a
new fixture, capture it from a real run (`raw.stdout` in any record) rather than typing
one from memory.

If you change or add a documented example in `docs/API.md`, `tests/test_docs_examples.py`
executes it. An example that no longer runs fails the suite rather than the reader.

## Adding an example config

Templates live in `configs/examples/`. A template:

* opens with a comment whose first line is the question it answers, followed by a
  paragraph on what it measures and why;
* says, in that header, the `mlsys doctor --config ...` check to run first and the
  `mlsys run ...` command itself;
* marks every machine-specific line (paths, hosts, ceilings) with `# <-- EDIT` and
  carries nothing from your own machines.

`tests/test_config_gallery.py` checks all of that. Then regenerate the gallery page
with `python tools/gen_config_gallery.py`; CI fails if `docs/configs.md` is stale.

## Documentation

The site at https://manunicholasjacob.com/ml-systems-lab/ is built from `docs/` with
MkDocs and deployed from `main` by `.github/workflows/docs.yml`.

```
pip install -e ".[docs]"
mkdocs serve              # live preview
mkdocs build --strict     # what CI runs
```

Pages that would repeat a file kept elsewhere (`results/README.md`,
`docker/README.md`, `CHANGELOG.md`, this file) include it with a snippet instead of
copying it, so edit the original. Links inside those files should be absolute URLs,
because they are rendered from two places.

## Adding a backend

A backend is three methods (`discover`, `build_task`, `parse`) against the contract in
`src/mlsyslab/backends/base.py`: one `RunSpec` in, one agent task out, one `RunRecord`
back. Ground rules:

* the agent returns raw output, the host parses it. Never parse on the device;
* every metric your backend cannot measure stays `None`. Never zero, never a guess;
* register it in `backends/__init__.py`, lazily if it imports anything heavy;
* add parser tests from a recorded output before wiring it to hardware.

## Adding a device kind

Subclass `Device` (`devices/base.py`), implement `_invoke_agent`, `push`, `pull`,
`resolve`, `exists`, `package_root`, and register it in `devices/__init__.py`. The agent
payload must reach the device as source and run under its Python with no third-party
imports; that constraint is what keeps every device equal.

## The agent image

`docker/Dockerfile` builds the image the Kubernetes device runs. A release publishes
a portable build (`GGML_NATIVE=OFF`, amd64 and arm64) to
`ghcr.io/manunicholasjacob/mlsyslab-agent` through
`.github/workflows/publish-image.yml`; a pull request that touches `docker/` builds it
without pushing. A local build keeps the default `GGML_NATIVE=ON`, which tunes
llama.cpp for the building machine: right for a benchmark, wrong for an image anyone
else will pull.

## Measurement changes

Anything touching the measurement path (agent, telemetry, backend argv) must respect the
rules in [docs/METHOD.md](https://github.com/manunicholasjacob/ml-systems-lab/blob/main/docs/METHOD.md), and a change that alters what a number means
must bump `SCHEMA_VERSION` in `schema.py`. Old records must remain loadable: `from_dict`
drops unknown fields by design, and there is a test that enforces it.

## Style

* Measurement core stays standard library; numpy/matplotlib/yaml only in `analysis/`
  and `config.py` (YAML), enforced by the no-deps CI job.
* Comments explain why, not what. The codebase leans on docstrings that record the
  failure a design decision prevents; keep that habit.
* No em dashes in any text, code comments included.
