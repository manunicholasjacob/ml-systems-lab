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
