# Recorded OpenVINO benchmark_app output

Raw stdout and stderr of OpenVINO `benchmark_app`, copied byte-for-byte from the run.
They exist so an OpenVINO parser (issue #3) can be written and tested against real
output instead of output typed from memory. Nothing here is parsed yet.

## What was run

- Model: `tiny_cnn.onnx`, rebuilt with `python tools/make_tiny_cnn_onnx.py tiny_cnn.onnx`
  (seed 0). Its SHA-256 is in `MANIFEST.json`; check it before comparing timings.
- Input shape `[1,3,32,32]` (the model has a dynamic batch dimension, so the shape is
  given on the command line).
- Three configurations, five reps each:
  - `cpu_sync_4threads`: `-d CPU -hint none -nstreams 1 -nthreads 4 -api sync -niter 2000`
  - `cpu_throughput_async`: `-d CPU -hint throughput -api async -t 10`
  - `gpu0_latency_sync`: `-d GPU.0 -hint latency -api sync -niter 2000` (Intel Iris Xe)
- Two deliberate failures, so the parser has recorded error paths (exit codes in the
  manifest): `cpu_missing_model` (a model path that does not exist, which fails in the
  ONNX frontend) and `cpu_hint_with_nthreads` (`-nthreads` together with `-hint latency`,
  which benchmark_app rejects while parsing arguments, before any device is opened).
- Each rep was interleaved with an ONNX Runtime run of the same model
  (`python -m mlsyslab.bench_onnx --threads 4 --batch 1`); odd reps ran OpenVINO first,
  even reps ONNX Runtime first. The ONNX Runtime output is not stored here.

`MANIFEST.json` lists every file with the exact argv, exit code, start and end time
(UTC), the OpenVINO build string and the host.

## Rules for these files

- Do not edit them. `tests/test_openvino_fixtures.py` checks every file against the
  SHA-256 in the manifest. `.gitattributes` marks the folder `-text` so Git does not
  rewrite the CRLF line endings `benchmark_app` writes on Windows.
- Read them through `openvino_fixture()` in that test module.
- The timings describe one laptop on one afternoon. They are test input for a parser,
  not a benchmark result, and should not be quoted as one.
- A parser should keep the project's rules: the host parses raw output, and a metric
  the output does not contain stays `None`, never zero.
