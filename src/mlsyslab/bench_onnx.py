"""ONNX Runtime benchmark, executed as a subprocess on the device under test.

This is the only module in the package that imports onnxruntime and numpy, and it is
deliberately a separate process from the agent. The agent stays standard library, so it
can run on a board with no scientific stack; this script runs only when an ONNX Runtime
run is actually requested, and only on a device that has the library.

It prints one JSON object on stdout. The host parses it. Like the llama.cpp path, nothing
is interpreted on the device.

Run it directly to sanity-check an install:

    python -m mlsyslab.bench_onnx --model model_int8.onnx --threads 4 --reps 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional

RESULT_BEGIN = "===MLSYSLAB-ONNX-BEGIN==="
RESULT_END = "===MLSYSLAB-ONNX-END==="


def _resolve_shape(dims: List[Any], batch: int, fallback: Dict[str, int]) -> List[int]:
    """Turn a possibly symbolic ONNX shape into concrete dimensions.

    Exported vision models routinely have a symbolic batch dimension and sometimes
    symbolic spatial dimensions. Guessing silently would produce a benchmark of the wrong
    input size, so unknown dimensions come from an explicit table and the resolved shape
    is reported back in the result.
    """
    resolved: List[int] = []
    for index, dim in enumerate(dims):
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        elif index == 0:
            resolved.append(batch)
        else:
            name = str(dim)
            resolved.append(int(fallback.get(name, fallback.get(str(index), 224))))
    return resolved


def build_session(args: argparse.Namespace):
    import onnxruntime as ort

    options = ort.SessionOptions()
    if args.threads:
        options.intra_op_num_threads = int(args.threads)
    # Inter-op parallelism is left at one thread: with a single sequential graph it adds
    # scheduling noise without adding throughput, and it makes the thread axis ambiguous.
    options.inter_op_num_threads = 1
    options.graph_optimization_level = {
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[args.optimization]
    if args.execution_mode == "sequential":
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    else:
        options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    providers = args.providers or ["CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    missing = [p for p in providers if p not in available]
    usable = [p for p in providers if p in available] or ["CPUExecutionProvider"]

    start = time.perf_counter()
    session = ort.InferenceSession(args.model, sess_options=options, providers=usable)
    load_ms = (time.perf_counter() - start) * 1000.0
    return session, load_ms, usable, missing


def run(args: argparse.Namespace) -> Dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    session, load_ms, providers, missing = build_session(args)

    fallback = json.loads(args.dim_overrides) if args.dim_overrides else {}
    feeds: Dict[str, Any] = {}
    input_report = []
    for tensor in session.get_inputs():
        shape = _resolve_shape(tensor.shape, args.batch, fallback)
        dtype = {
            "tensor(float)": np.float32, "tensor(float16)": np.float16,
            "tensor(double)": np.float64, "tensor(int64)": np.int64,
            "tensor(int32)": np.int32, "tensor(uint8)": np.uint8,
            "tensor(int8)": np.int8, "tensor(bool)": np.bool_,
        }.get(tensor.type, np.float32)
        if np.issubdtype(dtype, np.floating):
            data = np.random.rand(*shape).astype(dtype)
        else:
            data = np.zeros(shape, dtype=dtype)
        feeds[tensor.name] = data
        input_report.append({"name": tensor.name, "shape": shape, "type": tensor.type})

    output_names = [t.name for t in session.get_outputs()]

    # Warmup is not optional. The first call allocates arenas, resolves kernels and, on a
    # cold page cache, faults the weights in. Including it would measure the loader.
    for _ in range(max(0, args.warmup)):
        session.run(output_names, feeds)

    timings_ms: List[float] = []
    wall_start = time.perf_counter()
    for _ in range(max(1, args.reps)):
        start = time.perf_counter()
        session.run(output_names, feeds)
        timings_ms.append((time.perf_counter() - start) * 1000.0)
    wall_s = time.perf_counter() - wall_start

    array = np.array(timings_ms, dtype=np.float64)
    batch = args.batch or 1
    result = {
        "status": "ok",
        "onnxruntime_version": ort.__version__,
        "providers_used": providers,
        "providers_unavailable": missing,
        "model": args.model,
        "inputs": input_report,
        "threads": args.threads,
        "reps": len(timings_ms),
        "warmup": args.warmup,
        "load_ms": load_ms,
        "latency_ms_mean": float(array.mean()),
        "latency_ms_p50": float(np.percentile(array, 50)),
        "latency_ms_p95": float(np.percentile(array, 95)),
        "latency_ms_p99": float(np.percentile(array, 99)),
        "latency_ms_min": float(array.min()),
        "latency_ms_max": float(array.max()),
        "latency_ms_stdev": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "throughput_ips": (len(timings_ms) * batch) / wall_s if wall_s > 0 else None,
        "wall_s": wall_s,
    }
    if args.keep_samples:
        result["samples_ms"] = [round(v, 4) for v in timings_ms]
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlsyslab.bench_onnx")
    parser.add_argument("--model", required=True)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--providers", action="append", default=None)
    parser.add_argument("--optimization", default="all",
                        choices=["disabled", "basic", "extended", "all"])
    parser.add_argument("--execution-mode", default="sequential",
                        choices=["sequential", "parallel"])
    parser.add_argument("--dim-overrides", default=None,
                        help="JSON mapping of symbolic dimension names to sizes")
    parser.add_argument("--keep-samples", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write(RESULT_BEGIN + "\n")
    sys.stdout.write(json.dumps(result, default=str))
    sys.stdout.write("\n" + RESULT_END + "\n")
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
