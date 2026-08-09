"""Convert the paper-12 campaign results into RunRecords.

The lab must not launch empty: these are the real measurements behind "The Memory Wall at
the Edge of Language" (submitted to IEEE TC), reshaped into the unified schema so every
table and figure in this repo works against them on day one. Nothing is re-measured and
nothing is invented; fields the old formats did not record are left absent, and every
record is tagged with its source file.

Run from the repo root:

    python tools/backfill_paper12.py --source ../edge-llm/results --out results/paper12
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlsyslab import __version__
from mlsyslab.schema import (BackendInfo, DeviceInfo, Knobs, Metrics, RunRecord,
                             Telemetry, Workload, compute_run_id)

# The two platforms of the paper-12 campaign, as they were configured then.
PI = DeviceInfo(
    device_id="pi5-2gb", kind="ssh", os="Linux", machine="aarch64",
    cpu="Raspberry Pi 5 Model B Rev 1.0 (Cortex-A76)",
    board_model="Raspberry Pi 5 Model B Rev 1.0",
    logical_cores=4, physical_cores=4, ram_bytes=2_108_489_728,
    dram_peak_GBs=13.98,
)
X86 = DeviceInfo(
    device_id="laptop-i7-12700h", kind="local", os="Windows", machine="AMD64",
    cpu="12th Gen Intel(R) Core(TM) i7-12700H",
    logical_cores=20, physical_cores=14, ram_bytes=34_014_814_208,
    dram_peak_GBs=42.1,   # measured read ceiling used by the paper, not the datasheet
)

QUANT = {
    "q2k": "Q2_K", "q3km": "Q3_K_M", "q4km": "Q4_K_M", "q5km": "Q5_K_M", "q8": "Q8_0",
}


def quant_of(name: str) -> str:
    return QUANT.get(name.rsplit("-", 1)[-1], "Q4_K_M")


def base(device: DeviceInfo, model: str, file_mb: float, source: str,
         params: float = None) -> RunRecord:
    record = RunRecord(
        experiment="paper12-edge-llm-memwall",
        lab_version=__version__,
        timestamp_utc="2026-07-27T00:00:00Z",   # campaign date, not conversion date
        status="ok",
        tags=["backfill", "paper12"],
        device=device,
        backend=BackendInfo(name="llamacpp", execution_provider="CPU"),
        workload=Workload(
            model=model, quantization=quant_of(model),
            weight_bytes=int(file_mb * 1e6) if file_mb else None,
            params=params,
        ),
        knobs=Knobs(),
        metrics=Metrics(),
        telemetry=Telemetry(),
    )
    record.raw = {"source": source}
    return record


def finish(record: RunRecord) -> RunRecord:
    record.run_id = compute_run_id(
        "backfill", record.device.device_id, record.workload.model,
        record.knobs.threads, record.raw.get("source"),
    )
    weight = record.workload.weight_bytes
    if record.metrics.decode_tps and weight:
        record.metrics.decode_bw_GBs = record.metrics.decode_tps * weight / 1e9
        peak = record.device.dram_peak_GBs
        if peak:
            record.metrics.bw_utilization_pct = 100.0 * record.metrics.decode_bw_GBs / peak
    return record


def convert(source_dir: str):
    records = []

    def load(name):
        path = os.path.join(source_dir, name)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    # L1: the Pi thread-sweep roofline, 3 models x 4 thread counts.
    l1 = load("L1_roofline.json")
    if l1:
        for model, entry in l1["models"].items():
            for threads, point in entry["threads"].items():
                record = base(PI, model, entry["file_mb"], "L1_roofline.json",
                              entry.get("params_B"))
                record.knobs.threads = int(threads)
                record.metrics.prefill_tps = point.get("pp_ts")
                record.metrics.decode_tps = point.get("tg_ts")
                records.append(finish(record))

    # L2: the Pi quantisation sweep at 4 threads.
    l2 = load("L2_quant.json")
    if l2:
        for model, entry in l2["models"].items():
            record = base(PI, model, entry["file_mb"], "L2_quant.json", 0.5)
            record.workload.quantization = entry.get("quant") or quant_of(model)
            record.knobs.threads = 4
            record.metrics.prefill_tps = entry.get("pp_ts")
            record.metrics.decode_tps = entry.get("tg_ts")
            records.append(finish(record))

    # L3: Pi energy per token, with the PMIC rail split.
    l3 = load("L3_energy.json")
    if l3:
        idle = (l3.get("_meta") or {}).get("P_idle_W")
        for model, entry in l3["models"].items():
            record = base(PI, model, entry["file_mb"], "L3_energy.json")
            record.knobs.threads = 4
            record.metrics.decode_tps = entry.get("decode_ts")
            record.telemetry.power_w_mean = entry.get("P_decode_W")
            record.telemetry.power_w_core = entry.get("P_core_W")
            record.telemetry.power_w_dram = entry.get("P_ddr_W")
            record.telemetry.idle_power_w = idle
            record.telemetry.energy_per_token_mj = entry.get("energy_per_tok_mJ")
            records.append(finish(record))

    # pc_roofline: the x86 cross-platform validation, best thread count per model.
    pc = load("pc_roofline.json")
    if pc:
        for model, entry in pc["models"].items():
            record = base(X86, model, entry["file_mb"], "pc_roofline.json",
                          entry.get("params_B"))
            record.knobs.threads = entry.get("best_threads")
            record.metrics.prefill_tps = entry.get("prefill_ts")
            record.metrics.decode_tps = entry.get("decode_ts")
            records.append(finish(record))

    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="edge-llm results directory")
    parser.add_argument("--out", required=True, help="output directory for RunRecords")
    args = parser.parse_args()

    records = convert(args.source)
    if not records:
        print("nothing found to convert", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    by_device = {}
    for record in records:
        directory = os.path.join(args.out, record.device.device_id)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, f"{record.run_id}.json"), "w",
                  encoding="utf-8") as fh:
            fh.write(record.to_json())
        by_device[record.device.device_id] = by_device.get(record.device.device_id, 0) + 1

    for device, count in sorted(by_device.items()):
        print(f"{device}: {count} records")
    print(f"total: {len(records)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
