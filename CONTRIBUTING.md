# Contributing

## Running the tests

```
pip install -e ".[dev]"
python -m pytest -q
```

The suite is hardware-free: recorded llama-bench output and canned agent results stand
in for real devices, so it runs identically on a laptop and in CI. If a change needs a
new fixture, capture it from a real run (`raw.stdout` in any record) rather than typing
one from memory.

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

## Measurement changes

Anything touching the measurement path (agent, telemetry, backend argv) must respect the
rules in [docs/METHOD.md](docs/METHOD.md), and a change that alters what a number means
must bump `SCHEMA_VERSION` in `schema.py`. Old records must remain loadable: `from_dict`
drops unknown fields by design, and there is a test that enforces it.

## Style

* Measurement core stays standard library; numpy/matplotlib/yaml only in `analysis/`
  and `config.py` (YAML), enforced by the no-deps CI job.
* Comments explain why, not what. The codebase leans on docstrings that record the
  failure a design decision prevents; keep that habit.
* No em dashes in any text, code comments included.
