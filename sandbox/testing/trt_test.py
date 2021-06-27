from tvm import relay
import tvm
from tvm.relay.transform.backend_operator.utils import is_function_node
from tvm.relay.transform.backend_operator.target import measure, NUM_MEASUREMENTS_PER_REPEAT, NUM_REPEATS, AUTOTVM_LOG, AUTOSCH_LOG
from tvm.relay.transform.backend_operator.target import OPT_LEVEL
from tvm.relay.transform.optimizer.custom_fusion_pass import CustomFusionPass
from workloads.torch_workloads import get_network_from_torch
from tvm.contrib import graph_executor as runtime
import numpy as np

# from tensorrt.py
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import transform
from tvm.relay.expr_functor import ExprMutator, ExprVisitor

# visualize
from tvm.relay.transform.utility.visualize import visualize_network


class RemoveDropout(ExprMutator):
    """
    Removes all nn.dropout from an expr.
    """

    def visit_tuple_getitem(self, op):
        visit = super().visit_tuple_getitem(op)
        if visit.index != 0:
            return visit
        if (
            isinstance(visit.tuple_value, Call)
            and visit.tuple_value.op.name == "nn.dropout"
            and visit.index == 0
        ):
            return visit.tuple_value.args[0]
        return visit

@transform.function_pass(opt_level=0)
class RemoveDropoutPass:
    def transform_function(self, func, mod, _):
        return RemoveDropout().visit(func)



def print_ir(mod, info, is_before):
    """Print the name of the pass, the IR, only before passes execute."""
    if info.name == "AnnotateTargetFunc" or info.name == "MergeCompilerRegions" or info.name == "PartitionGraph":
        if is_before:
            print("Running pass: {}", info.name)
        #print(mod)
        else:
            print("Done pass: {}", info.name)
            visualize_network(mod["main"], info.name)




def partition_for_tensorrt(
    mod,
    params=None,
    version=None,
    use_implicit_batch=True,
    remove_no_mac_subgraphs=False,
    max_workspace_size=1 << 30,
):
    """Partition the graph greedily offloading supported operators to TensorRT.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.
    version : Optional[Tuple[int, int, int]]
        TensorRT version to target as tuple of (major, minor, patch). If TVM is compiled with
        USE_TENSORRT_RUNTIME=ON, the linked TensorRT version will be used instead.
    use_implicit_batch : Optional[bool]
        Use TensorRT implicit batch mode (default true). Setting to false will enable explicit batch
        mode which will widen supported operators to include those which modify the batch dimension,
        but may reduce performance for some models.
    remove_no_mac_subgraphs : Optional[bool]
        Removes subgraphs which have been partitioned for TensorRT if they do not have any
        multiply-accumulate operations. The removed subgraphs will go through TVM's standard
        compilation instead. Can improve performance.
    max_workspace_size : Optional[int]
        How many bytes of workspace size to allow each subgraph to use for TensorRT engine creation.
        See TensorRT documentation for more info.
    Returns
    -------
    mod_and_config : Tuple[Module, Dict[str, Any]]
        A tuple of 1) annotated and partitioned module and 2) "relay.ext.tensorrt.options"
        configuration which should be given to PassContext when building.
    """
    config = {
        "use_implicit_batch": use_implicit_batch,
        "max_workspace_size": max_workspace_size,
        "remove_no_mac_subgraphs": remove_no_mac_subgraphs,
    }
    if version:
        assert isinstance(version, tuple) and len(version) == 3
        config["tensorrt_version"] = version
    else:
        linked_version = tuple(tvm.get_global_func("relay.op.get_tensorrt_version")())
        if not linked_version:
            logger.warning(
                "TVM was not built against TensorRT and no version was provided to "
                "partition_for_tensorrt. Defaulting to 6.0.1"
            )
            linked_version = (6, 0, 1)
        config["tensorrt_version"] = linked_version

    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)
    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            RemoveDropoutPass(),
            transform.RemoveUnusedFunctions(),
            transform.ConvertLayout(
                {
                    "nn.conv2d": ["NCHW", "default"],
                    "nn.conv3d": ["NCDHW", "default"],
                    "nn.conv2d_transpose": ["NCHW", "default"],
                }
            ),
            transform.FoldConstant(),
            transform.AnnotateTarget("tensorrt"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
            transform.InferType(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3, config={"relay.ext.tensorrt.options": config}, trace=print_ir):
        mod = seq(mod)
        #mod = prune_tensorrt_subgraphs(mod)

    return mod, config


def measure_end_to_end_perf_tensorrt(mod, params, target_str, shape_dict, is_ours):
    #from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
    mod, config = partition_for_tensorrt(mod, params)

    with tvm.transform.PassContext(opt_level=OPT_LEVEL.get(), config={'relay.ext.tensorrt.options': config}):
        lib = relay.build(mod, target=target_str, params=params)

    lib.export_library('compiled.so')

    dev = tvm.gpu(0)
    loaded_lib = tvm.runtime.load_module('compiled.so')
    module = tvm.contrib.graph_executor.GraphModule(loaded_lib['default'](dev))

    # Setup execution
    for input_name, input_shape in shape_dict.items():
        input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
        module.set_input(input_name, input_data)

    ftimer = module.module.time_evaluator("run", dev, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)

    return measure(ftimer, is_net=True)


if __name__ == "__main__":
    network = "resnet50"

    # We can't test this because this network include batch norm.
    mod, params, shape_dict, _ = get_network_from_torch(network, 1)

    mean_perf, std_perf = measure_end_to_end_perf_tensorrt(mod, params, 'cuda', shape_dict, False)
    print(f"[TensorRT] Performance of {network} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")


