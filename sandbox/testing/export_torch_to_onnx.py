import argparse
import torch
from workloads.workloads import WORKLOADS_DIC
from workloads.torch_workloads import load_torch_model_from_code
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--network", help="name of a neural network")
    parser.add_argument("-bs", "--batch-size", default=1, type=int, help="batch size")
    args = parser.parse_args()

    input_names = ["input0"]
    output_names = ["output0"]

    net_to_output_shape = {
        "bert_full": (args.batch_size, 64, 256),
        "yolov3": (args.batch_size, 10647, 25),
    }

    scripted_model = load_torch_model_from_code(args.network, args.batch_size)
    input_shape = WORKLOADS_DIC[args.network][args.batch_size]["input0"]
    input_data = torch.randn(input_shape)

    this_code_path = os.path.dirname(os.path.abspath(__file__))
    onnx_path = f"{this_code_path}/baselines/onnx/{args.network}.onnx"

    torch.onnx.export(scripted_model, input_data,
                      onnx_path, verbose=False,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=input_names, output_names=output_names,
                      training=torch.onnx.TrainingMode.EVAL,
                      example_outputs=torch.randn(net_to_output_shape[args.network]),
                      opset_version=12)

    import onnx
    from onnx_tf.backend import prepare

    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)

    onnx_tf_path = f"{this_code_path}/baselines/onnx/{args.network}.pb"
    tf_rep.export_graph(onnx_tf_path)