from tvm import relay
import torch

import os
import copy

from .workloads import *
from baselines.pytorch.resnets import resnet50, resnext50_32x4d, resnet_block
from baselines.pytorch.nasnet_a import NASNetA
from baselines.pytorch.nasrnn import NASRNN
from baselines.pytorch.bert import BERT

NETWORK_TO_TORCH_MODEL = {
    "resnet_block": resnet_block,
    "resnet50" : resnet50,
    "resnext50_32x4d" : resnext50_32x4d,
    "nasneta" : NASNetA,
    "nasrnn": NASRNN,
    "bert": BERT,
}

# Warning(@Soo): It does not work for NasRNN
def load_torch_model_from_pth(name):
    # Get the model
    this_code_path = os.path.dirname(os.path.abspath(__file__))
    model = torch.jit.load(f"{this_code_path}/../baselines/pytorch/models/{name}.pth")
    model.eval()

    # Create the input data
    shape_dict = WORKLOADS_DIC[name]
    assert len(shape_dict) == 1
    for shape in shape_dict.values():
        input_shape = shape
    input_data = torch.randn(input_shape)

    scripted_model = torch.jit.trace(model.cpu(), input_data).eval()
    return scripted_model

def load_torch_model_from_code(name):
    # Get the model
    if name == "nasrnn":
        model = NETWORK_TO_TORCH_MODEL[name](is_gpu=False)#.cuda()
    else:
        model = NETWORK_TO_TORCH_MODEL[name]()  # .cuda()

    model.eval()

    # Create the input data
    shape_dict = WORKLOADS_DIC[name]
    assert len(shape_dict) == 1
    for shape in shape_dict.values():
        input_shape = shape
    input_data = torch.randn(input_shape)

    scripted_model = torch.jit.trace(model.cpu(), input_data).eval()
    return scripted_model

# NOTE: Make sure that you executed codes in "baselines/pytorch_new" to have the most recent onnx files
def get_network_from_torch(name, batch_size):
    assert name in WORKLOADS_DIC
    # if batch_size > 1, we need to think about how to take care of bert and nasrnn
    assert batch_size == 1

    torch_model = load_torch_model_from_code(name)
    # Set the input shape dict
    shape_dict = WORKLOADS_DIC[name]

    # Warning: For from_pytorch, we should convert dictionary to list
    shape_arr = list(shape_dict.items())
    # We should copy shape_dict because shape_dict will be consumed in from_onnx
    shape_arr_tmp = copy.deepcopy(shape_arr)
    mod, params = relay.frontend.from_pytorch(torch_model, shape_arr_tmp)#, freeze_params=True)

    print(f"(Loaded network, Shape array) = ({name}, {shape_arr})")
    return mod, params, shape_dict, None # we don't need output shape

def crop_network_from_torch(name, batch_size, post_dfs_order):
    mod, params, shape_dict, _ = get_network_from_torch(name, batch_size)
    expr = ExprCropper(post_dfs_order).crop(mod["main"])
    mod, params = create_relay_workload(expr)

    return mod, params, shape_dict, None
