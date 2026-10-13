**What this changes, and why:**

**How it was checked:**
- [ ] `python -m pytest -q` passes
- [ ] if it touches the measurement path (agent, telemetry, backend argv): it follows
      docs/METHOD.md, and `SCHEMA_VERSION` is bumped if a number's meaning changed
- [ ] if it adds or edits a config: `python tools/gen_config_gallery.py` was re-run
- [ ] if it changes docs: `mkdocs build --strict` passes
- [ ] for a new backend or device: tested on real hardware (say which), with a recorded
      fixture added for the tests

**Records or output**, if the change affects what a run produces:
