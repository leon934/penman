import numpy as np
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
from onnx import TensorProto, version_converter
import pickle

from convolution import Convolution
from activation import Sigmoid, ReLU, Softmax
from reshape import Reshape
from dense import Dense
from pooling import Pooling

weights = []
nodes = []
inputs = []
outputs = []

IR_VERS = 10
OPT_VERS = 15

def main():
    num_output = 17

    with open("./model/model.pkl", "rb") as file:
        layers = pickle.load(file)

    for i, layer in enumerate(layers):
        if isinstance(layer, Convolution):
            kernels = layer.kernels
            bias = layer.bias

            weights.append(numpy_helper.from_array(kernels.astype(np.float32), name=f"conv{i}_kernel"))
            weights.append(numpy_helper.from_array(bias.astype(np.float32), name=f"conv{i}_bias"))

            nodes.append(helper.make_node(
                    "Conv",
                    inputs=["X", f"conv{i}_kernel"],
                    outputs=[f"conv{i}_no_bias_out"],
                    pads=[0, 0, 0, 0],
                    strides=[1, 1]
                )
            )

            nodes.append(helper.make_node(
                    "Add",
                    inputs=[f"conv{i}_no_bias_out", f"conv{i}_bias"],
                    outputs=[f"conv{i}_out"]
                )
            )

            prev = f"conv{i}_out"
        elif isinstance(layer, ReLU):
            nodes.append(helper.make_node(
                    "Relu",
                    inputs=[prev],
                    outputs=[f"relu{i}_out"]
                )
            )

            prev = f"relu{i}_out"
        elif isinstance(layer, Pooling):
            nodes.append(helper.make_node(
                    "MaxPool",
                    inputs=[prev],
                    outputs=[f"pool{i}_out"],
                    kernel_shape=[2, 2],
                    strides=[2, 2]
                )
            )

            prev = f"pool{i}_out"
        elif isinstance(layer, Reshape):
            nodes.append(helper.make_node(
                    "Flatten",
                    inputs=[prev],
                    outputs=[f"flat{i}_out"],
                    axis=1
                )
            )

            prev = f"flat{i}_out"
        elif isinstance(layer, Dense):
            d_weights = layer.weights.T

            # Flattens to shape (output_size,) instead of (output_size, 1)
            d_bias = layer.bias.reshape(-1)

            weights.append(numpy_helper.from_array(d_weights.astype(np.float32), f"gemm{i}_weight"))
            weights.append(numpy_helper.from_array(d_bias.astype(np.float32), f"gemm{i}_bias"))

            nodes.append(helper.make_node(
                    "Gemm",
                    inputs=[prev, f"gemm{i}_weight", f"gemm{i}_bias"],
                    outputs=[f"gemm{i}_out"],
                    alpha=1.0, beta=1.0, transB=0
                )
            )

            prev = f"gemm{i}_out"
        elif isinstance(layer, Softmax):
            nodes.append(helper.make_node(
                    "Softmax",
                    inputs=[prev],
                    outputs=["Y"],
                    axis=1
                )
            )

    graph_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, *layers[0].input_size])
    graph_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, num_output])

    graph = helper.make_graph(
        nodes,
        "cnn_graph",
        inputs=[graph_input],
        outputs=[graph_output],
        initializer=weights
    )

    model = helper.make_model(graph, producer_name="penman_cnn")
    model.ir_version = IR_VERS
    model = version_converter.convert_version(model, OPT_VERS)

    onnx.checker.check_model(model)
    onnx.save(model, "./model/penman_cnn.onnx")

if __name__ == "__main__":
    main()