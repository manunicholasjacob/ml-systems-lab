"""ml-systems-lab: reproducible ML inference benchmarking across heterogeneous hardware.

The package is layered so that each piece can be tested without the layer below it:

    config   -> expands one YAML file into a list of concrete RunSpecs
    devices  -> where a run executes (this machine, or a Pi over SSH)
    backends -> what runs (llama.cpp, ONNX Runtime), and how to parse its output
    agent    -> the stdlib-only payload that executes on the device and samples telemetry
    schema   -> the single RunRecord every run produces, whatever device or backend
    analysis -> a directory of RunRecords in, tables and figures out

Only ``analysis`` requires numpy or matplotlib. Everything on the measurement path is
standard library, because it has to run on a 2 GB Raspberry Pi that we do not want to
install a scientific stack onto.
"""

__version__ = "0.1.1"

__all__ = ["__version__"]
