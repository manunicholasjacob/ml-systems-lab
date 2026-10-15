"""Write the small ONNX model that the recorded OpenVINO output was produced with.

The recordings under tests/recorded/openvino/ need a model anyone can rebuild, so this
script builds one from a fixed seed instead of pointing at a checkpoint that lives on
one machine. It is a plain CIFAR-sized CNN (two convolutions, global pooling, a linear
layer, 10 classes) with random weights. The weights do not matter for a timing run;
the graph shape does, and it is fixed here.

    python tools/make_tiny_cnn_onnx.py tiny_cnn.onnx

Needs `onnx` and `numpy`. The same onnx version writes the same bytes; the SHA-256 of
the file used for the recordings is in tests/recorded/openvino/MANIFEST.json.
"""

import argparse
import hashlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(seed: int = 0) -> onnx.ModelProto:
    rng = np.random.default_rng(seed)

    def w(name, *shape):
        return numpy_helper.from_array(
            (rng.standard_normal(shape) * 0.1).astype(np.float32), name)

    inits = [
        w("conv1_w", 32, 3, 3, 3), w("conv1_b", 32),
        w("conv2_w", 64, 32, 3, 3), w("conv2_b", 64),
        w("fc_w", 10, 64), w("fc_b", 10),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "conv1_w", "conv1_b"], ["c1"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["r1"]),
        helper.make_node("Conv", ["r1", "conv2_w", "conv2_b"], ["c2"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2]),
        helper.make_node("Relu", ["c2"], ["r2"]),
        helper.make_node("GlobalAveragePool", ["r2"], ["gap"]),
        helper.make_node("Flatten", ["gap"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "fc_w", "fc_b"], ["logits"], transB=1),
    ]
    graph = helper.make_graph(
        nodes, "tiny_cnn",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 32, 32])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 10])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                              producer_name="ml-systems-lab")
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    data = build(args.seed).SerializeToString()
    with open(args.out, "wb") as f:
        f.write(data)
    print(f"{args.out}  {len(data)} bytes  sha256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
