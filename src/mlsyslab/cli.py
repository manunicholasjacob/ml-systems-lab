"""Command line entry point.

    mlsys probe   --config configs/lab.yaml     what the machines are and can measure
    mlsys run     configs/laptop-smoke.yaml     execute a config, resumably
    mlsys report  runs/laptop-smoke             tables from a results directory

``run`` is the only command that touches hardware. ``probe`` is safe to run at any time
and is the first thing to try when a config does not behave, because it prints the
capability map that decides which telemetry a device can actually produce.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from typing import List, Optional

from . import __version__


def _cmd_probe(args: argparse.Namespace) -> int:
    from . import devices as device_registry

    targets = {}
    if args.config:
        from .config import load

        config = load(args.config)
        targets = config.devices
    if args.device:
        targets = {k: v for k, v in targets.items() if k in args.device}
        for name in args.device:
            if name not in targets:
                targets[name] = {"kind": "local"} if name == "local" else {"host": name}
    if not targets:
        targets = {"local": {"kind": "local"}}

    payload = {}
    for device_id, device_config in targets.items():
        try:
            device = device_registry.from_config(device_id, device_config)
            payload[device_id] = {
                "device": device.device_info(),
                "capabilities": device.capabilities(),
            }
        except Exception as exc:
            payload[device_id] = {"error": f"{type(exc).__name__}: {exc}"}

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0

    for device_id, entry in payload.items():
        print(f"\n{device_id}")
        print("-" * len(device_id))
        if "error" in entry:
            print(f"  unreachable: {entry['error']}")
            continue
        info = entry["device"]
        for key in ("cpu", "board_model", "machine", "distro", "kernel", "compiler",
                    "logical_cores", "physical_cores", "ram_bytes", "swap_bytes",
                    "gpu", "python", "dram_peak_GBs"):
            if info.get(key) is not None:
                value = info[key]
                if key.endswith("_bytes"):
                    value = f"{value / 1e9:.2f} GB"
                print(f"  {key:<16} {value}")
        can = [k for k, v in entry["capabilities"].items() if v]
        cannot = [k for k, v in entry["capabilities"].items() if not v]
        print(f"  {'can measure':<16} {', '.join(can) or 'nothing'}")
        print(f"  {'cannot':<16} {', '.join(cannot) or 'nothing'}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .config import load, specs_to_table
    from .runner import Runner, console_reporter

    config = load(args.config)
    specs = config.specs
    if args.device:
        specs = [s for s in specs if s.device_id in args.device]
    if args.model:
        specs = [s for s in specs if s.model in args.model]
    if args.mode:
        specs = [s for s in specs if s.mode == args.mode]
    if args.limit:
        specs = specs[: args.limit]

    if not specs:
        sys.stderr.write("no runs match the given filters\n")
        return 1

    print(f"{config.experiment}: {len(specs)} run(s)")
    print(specs_to_table(specs))

    if args.dry_run:
        print("\ndry run, nothing executed")
        return 0

    runner = Runner(config, output_dir=args.output, resume=not args.no_resume,
                    on_event=console_reporter())
    print(f"\nwriting to {runner.output_dir}\n")
    records = runner.run(specs)
    return 1 if any(not r.ok for r in records) else 0


def _cmd_report(args: argparse.Namespace) -> int:
    from .schema import load_records

    if args.json:
        records = load_records(args.directory)
        if not records:
            sys.stderr.write(f"no run records found under {args.directory}\n")
            return 1
        print(json.dumps([r.to_dict() for r in records], indent=2, default=str))
        return 0

    from .analysis import Dataset, report, tables

    if args.full:
        path = report(args.directory, output_dir=args.output,
                      table_format=args.format if args.format != "text" else "markdown")
        print(f"report written to {path}")
        return 0

    dataset = Dataset.from_directory(args.directory)
    if not len(dataset):
        sys.stderr.write(f"no successful run records under {args.directory}\n")
        return 1
    for note in dataset.integrity_warnings():
        sys.stderr.write(f"note: {note}\n")
    for _title, builder in tables.ALL_TABLES:
        rendered = builder(dataset, fmt=args.format)
        if not rendered.startswith("("):
            print(rendered)
            print()
    return 0


def _cmd_membw(args: argparse.Namespace) -> int:
    """Measure the DRAM read ceiling on a device, for its dram_peak_GBs config entry."""
    from . import devices as device_registry
    from .membw import RESULT_BEGIN, RESULT_END

    if not args.device or args.device == "local":
        from .membw import measure

        result = measure(working_set_mb=args.mb, reps=args.reps)
    else:
        if not args.config:
            sys.stderr.write("--device needs --config to know how to reach it\n")
            return 1
        from .config import load

        config = load(args.config)
        if args.device not in config.devices:
            sys.stderr.write(f"unknown device '{args.device}' in {args.config}\n")
            return 1
        device = device_registry.from_config(args.device, config.devices[args.device])
        agent_result = device.execute({
            "kind": "command",
            "argv": [device.python_executable, "-m", "mlsyslab.membw",
                     "--reps", str(args.reps)] + (["--mb", str(args.mb)] if args.mb else []),
            "env": {"PYTHONPATH": device.package_root()},
            "timeout_s": 900,
            "sample_power": False,
        })
        stdout = agent_result.get("stdout") or ""
        start, end = stdout.find(RESULT_BEGIN), stdout.find(RESULT_END)
        if start == -1 or end == -1:
            sys.stderr.write(f"membw failed on {args.device}: "
                             f"{agent_result.get('error')}\n{stdout[-500:]}\n")
            return 1
        result = json.loads(stdout[start + len(RESULT_BEGIN):end].strip())

    if result.get("status") != "ok":
        sys.stderr.write(f"failed: {result.get('error')}\n")
        return 1
    print(f"peak read bandwidth : {result['peak_read_GBs']:.2f} GB/s "
          f"({result['best_kernel']} kernel, {result['best_threads']} threads)")
    print(f"copy (for context)  : {result['copy_GBs']:.2f} GB/s")
    print(f"working set         : {result['working_set_mb']} MB, {result['reps']} reps")
    if result.get("stability") is not None:
        print(f"stability           : median/best = {result['stability']:.2f}")
    if result.get("warning"):
        print(f"\nWARNING: {result['warning']}")
    print(f"\nput this in the device config:\n  dram_peak_GBs: {result['peak_read_GBs']:.2f}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Check that the local environment is ready to run experiments."""
    import shutil

    from . import devices as device_registry

    ok = True

    print("ml-systems-lab doctor")
    print("=" * 40)

    print(f"\n  Python        : {sys.executable} ({platform.python_version()})")
    print(f"  Platform      : {platform.platform()}")
    print(f"  Lab version   : {__version__}")

    # Check optional dependencies
    print("\nDependencies:")
    for pkg, extra in [("yaml", "PyYAML (required for YAML configs)"),
                       ("numpy", "numpy (required for analysis)"),
                       ("matplotlib", "matplotlib (required for figures)"),
                       ("onnxruntime", "onnxruntime (optional, ORT backend)")]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"  [ok] {extra} {ver}")
        except ImportError:
            severity = "MISSING" if pkg in ("yaml",) else "skip"
            if severity == "MISSING":
                ok = False
            print(f"  [{severity}] {extra}")

    # Check llama.cpp binaries
    print("\nllama.cpp:")
    for binary in ("llama-bench", "llama-server", "llama-cli"):
        path = shutil.which(binary)
        if path:
            print(f"  [ok] {binary} -> {path}")
        else:
            print(f"  [skip] {binary} not on PATH")

    # Check config if provided
    if args.config:
        print(f"\nConfig: {args.config}")
        try:
            from .config import load
            config = load(args.config)
            print(f"  experiment : {config.experiment}")
            print(f"  devices    : {', '.join(config.devices)}")
            print(f"  models     : {len(config.models)}")
            print(f"  run specs  : {len(config.specs)}")

            for name, model_cfg in config.models.items():
                for dev_id, path in (model_cfg.get("paths") or {}).items():
                    exists = os.path.isfile(path)
                    tag = "ok" if exists else "MISSING"
                    if not exists:
                        ok = False
                    print(f"  [{tag}] {name} @ {dev_id}: {path}")

            for dev_id, dev_cfg in config.devices.items():
                try:
                    device = device_registry.from_config(dev_id, dev_cfg)
                    info = device.device_info()
                    print(f"  [ok] device '{dev_id}': {info.get('cpu', '?')}")
                except Exception as exc:
                    print(f"  [FAIL] device '{dev_id}': {exc}")
                    ok = False
        except Exception as exc:
            print(f"  [FAIL] could not load: {exc}")
            ok = False

    print()
    if ok:
        print("All checks passed.")
    else:
        print("Some checks failed; see [MISSING] / [FAIL] above.")
    return 0 if ok else 1


def _cmd_compare(args: argparse.Namespace) -> int:
    """Same workload measured in two result sets, side by side with the ratio."""
    from .analysis import Dataset
    from .analysis.tables import render

    left = Dataset.from_directory(args.left)
    right = Dataset.from_directory(args.right)
    if not len(left) or not len(right):
        sys.stderr.write("one of the directories has no successful records\n")
        return 1

    def key(row):
        return (row["backend"], row["model"], row["quant"], row["mode"],
                row["threads"], row["prompt_tokens"], row["output_tokens"],
                row["context_tokens"], row["batch"])

    metric = args.metric
    left_best, right_best = {}, {}
    for rows, best in ((left.rows, left_best), (right.rows, right_best)):
        for row in rows:
            if row.get(metric) is None:
                continue
            k = key(row)
            if k not in best or row[metric] > best[k][metric]:
                best[k] = row

    shared = sorted(set(left_best) & set(right_best), key=lambda k: str(k))
    if not shared:
        sys.stderr.write(f"no workload with '{metric}' appears in both directories\n")
        return 1

    lower_is_better = metric in ("ttft_ms", "e2e_ms", "latency_ms", "load_ms")
    headers = ["model", "mode", "thr", f"{metric} A", f"{metric} B", "B/A"]
    rows = []
    for k in shared:
        a, b = left_best[k][metric], right_best[k][metric]
        rows.append([left_best[k]["model"], left_best[k]["mode"],
                     left_best[k]["threads"], a, b, b / a if a else None])
    caption = (f"A = {args.left}, B = {args.right}; best '{metric}' per workload"
               + ("; lower is better" if lower_is_better else ""))
    print(render(headers, rows, args.format, caption=caption))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlsys",
        description="Reproducible ML inference benchmarking across heterogeneous hardware.",
    )
    parser.add_argument("--version", action="version", version=f"ml-systems-lab {__version__}")
    sub = parser.add_subparsers(dest="command")

    probe = sub.add_parser("probe", help="describe devices and what they can measure")
    probe.add_argument("--config", help="config file whose devices should be probed")
    probe.add_argument("--device", action="append", help="probe only this device (repeatable)")
    probe.add_argument("--json", action="store_true", help="machine-readable output")
    probe.set_defaults(func=_cmd_probe)

    run = sub.add_parser("run", help="execute an experiment config")
    run.add_argument("config", help="path to a YAML or JSON experiment config")
    run.add_argument("--output", help="results directory (defaults to the config's output_dir)")
    run.add_argument("--dry-run", action="store_true", help="print the matrix and stop")
    run.add_argument("--no-resume", action="store_true", help="re-run points already on disk")
    run.add_argument("--device", action="append", help="restrict to this device (repeatable)")
    run.add_argument("--model", action="append", help="restrict to this model (repeatable)")
    run.add_argument("--mode", choices=["throughput", "latency"], help="restrict to one mode")
    run.add_argument("--limit", type=int, help="stop after this many runs")
    run.set_defaults(func=_cmd_run)

    report = sub.add_parser("report", help="summarise a results directory")
    report.add_argument("directory", help="directory of run records")
    report.add_argument("--json", action="store_true", help="dump the records as JSON")
    report.add_argument("--full", action="store_true",
                        help="write REPORT.md plus figures instead of printing tables")
    report.add_argument("--format", choices=["text", "markdown", "latex"], default="text",
                        help="table format (default: text)")
    report.add_argument("--output", help="output directory for --full (default: <dir>/report)")
    report.set_defaults(func=_cmd_report)

    membw = sub.add_parser("membw",
                           help="measure a device's DRAM read ceiling (dram_peak_GBs)")
    membw.add_argument("--device", help="device id from --config; omit for this machine")
    membw.add_argument("--config", help="config file defining the device")
    membw.add_argument("--mb", type=int, help="working set in MB")
    membw.add_argument("--reps", type=int, default=5)
    membw.set_defaults(func=_cmd_membw)

    compare = sub.add_parser("compare", help="compare a metric across two result dirs")
    compare.add_argument("left", help="results directory A")
    compare.add_argument("right", help="results directory B")
    compare.add_argument("--metric", default="decode_tps",
                         help="flattened column to compare (default decode_tps)")
    compare.add_argument("--format", choices=["text", "markdown", "latex"],
                         default="text")
    compare.set_defaults(func=_cmd_compare)

    doctor = sub.add_parser("doctor", help="check that the environment is ready")
    doctor.add_argument("--config", help="also validate a config file's paths and devices")
    doctor.set_defaults(func=_cmd_doctor)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted; completed runs are already on disk\n")
        return 130
    except Exception as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        if os.environ.get("MLSYSLAB_TRACEBACK"):
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
