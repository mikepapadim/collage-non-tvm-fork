import tvm._ffi
import tvm.driver
# import tvm.relay.testing as testing
# from tvm import relay

# from ..backend_operator.record import backendop_lib
from ..backend_operator.op_config import MeasuredConfigs
from ..backend_operator.target import Target, USER_DEFINED_MATCH_LOG
from ..backend_operator.backend_op_lib import BackendOpLib
from ..backend_operator.utils import *

#from ..utility.visualize import visualize_network
from ..utility.profile_ops_in_net import profile_ops_in_net
# from .optimizer_utils import *

from .comp_graph import ComputationGraph
from .comp_graph_optimizer import *
from .backend_op_state import *
from .evolutionary_searcher import EvolutionarySearcher
from .op_match_logger import *
from ..utility.debug_helper import printe
# from .ext_compiler_op_merger import *
from .ext_compiler_op_annotator import ExtCompilerOpAnnotator
from tvm.relay.op.contrib.tensorrt import prune_tensorrt_subgraphs
from tvm.relay import transform

def setup_backend_op_lib(network_expr, targets, batch_size, hw_name):
    backendop_lib = BackendOpLib.get(hw_name)
    # backendop_lib.measure_backend_ops(network_expr, targets, batch_size)

    return backendop_lib

@tvm._ffi.register_func("relay.transform.optimizer.print_attr_args")
def print_attr_args(expr):
    printe(f"attr: {get_attr_vals(expr)}")

@tvm._ffi.register_func("relay.transform.optimizer.visualize_network_debug")
def visualize_network_debug(relay_expr, name):
    net_name = 'default'
    if relay_expr.attrs is not None and NETWORK_FUNC_ATTR in dict(relay_expr.attrs):
        net_name = dict(relay_expr.attrs)[NETWORK_FUNC_ATTR]
        visualize_network(relay_expr, f"{net_name}_{name}")
        printe("[Done] Debug visualization")

@tvm._ffi.register_func("relay.transform.optimizer.apply_external_compiler_op")
def apply_external_compiler_op(mod):
    printe("External compiler op pass")

    # Get best op match info
    fn_body = mod["main"].body
    # print(f"backend body (before): {fn_body.backend}")
    # opt_match = OpMatchReader().read(fn_body)
    # print(f"backend body (after): {fn_body.backend}")
    # visualize_network(mod["main"], "notepad")
    # Annotating expression
    target_str = "tensorrt"
    # visualize_network(mod["main"], "AnnotateTargetFunc_before")
    # mod["main"] = ExtCompilerOpAnnotator(opt_match).annotate(mod["main"], target_str)
    # visualize_network(mod["main"], "AnnotateTargetFunc_after")

    # Do merge and partition pass
    use_implicit_batch = True
    remove_no_mac_subgraphs = False
    max_workspace_size = 1 << 30
    version = None

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

    # Warning(@Soo): I assume this is only useful when folding constant
    # if params:
    #     mod["main"] = bind_params_by_name(mod["main"], params)
    # printe("*" * 30)
    # printe("*" * 30)
    # printe("*" * 30)
    #
    # # backend exists
    # printe(f"Rerp(Python): {repr(fn_body)}")

    seq = tvm.transform.Sequential(
        [
            # transform.InferType(),
            # RemoveDropoutPass(),
            # transform.RemoveUnusedFunctions(),
            # transform.ConvertLayout(
            #     {
            #         "nn.conv2d": ["NCHW", "default"],
            #         "nn.conv3d": ["NCDHW", "default"],
            #         "nn.conv2d_transpose": ["NCHW", "default"],
            #     }
            # ),
            # transform.FoldConstant(),
            transform.AnnotateTarget("tensorrt"),
            transform.MergeCompilerRegions(),
            #tvm.ir.transform.PrintIR("After merging graph"),
            transform.PartitionGraph(),
            #tvm.ir.transform.PrintIR("After partitioning graph"),
            transform.InferType(),
        ]
    )

    # Do prune_tensorrt_subgraphs
    # with tvm.transform.PassContext(opt_level=OPT_LEVEL.get(), config={"relay.ext.tensorrt.options": config},trace=print_ir):
    with tvm.transform.PassContext(opt_level=OPT_LEVEL.get(), config={"relay.ext.tensorrt.options": config}):
        # printe("Before sequential")
        # printe(repr(mod["main"]))
        mod = seq(mod)
        # printe("After sequential")
        # Warning(@Soo): Would it be problematic?
        # mod = prune_tensorrt_subgraphs(mod)

    return mod
    # return mod, config

# For annotate_test
def get_temp_opt_match(relay_expr):
    printe("update backend from Python side")
    relay.analysis.update_backend(relay_expr, "0-tvmgpu-autotvm_add")
    relay.analysis.update_backend(relay_expr.args[0], "1-tvmgpu-autotvm_relu")
    relay.analysis.update_backend(relay_expr.args[1], "2-tensorrt_tanh")
    relay.analysis.update_backend(relay_expr.args[0].args[0],"3-tvmgpu-autotvm_relu")
    relay.analysis.update_backend(relay_expr.args[0].args[0].args[0], "3-tvmgpu-autotvm_relu")
    return relay_expr

@tvm._ffi.register_func("relay.transform.optimizer.get_user_fusion")
def get_user_fusion(relay_expr):
    printe("User-defined fusion")
    net_name = relay_expr.attrs[NETWORK_FUNC_ATTR]
    hw_name = relay_expr.attrs[HW_FUNC_ATTR]
    relay_expr = get_function_body(relay_expr)
    match_path = f"{LOG_PATH}/user_defined_match_{net_name}_{hw_name}.log"
    # match_path = f"{LOG_PATH}/best_match_{net_name}.log"
    opt_match = OpMatchReader().read(relay_expr, match_path)

    # # printe(repr(relay_expr))
    # if relay_expr.backend == 'default':
    #     opt_match = get_temp_opt_match(relay_expr)
    # visualize_network(relay_expr, "notepad")
    # return relay_expr

def run_op_level_opt(relay_expr):
    hw_name = relay_expr.attrs[HW_FUNC_ATTR]
    relay_expr = get_function_body(relay_expr)

    print(f"[Op-Level: DP] Computation graph generation...")
    comp_graph = ComputationGraph(relay_expr)
    n_relay_nodes = comp_graph.n_relay_nodes
    print(f"# of relay nodes in comp graph: {n_relay_nodes}")

    # Sanity check: Only AutoTVM
    # targets = [Target.TVM_GPU_AUTOTVM]
    # targets = [Target.TENSORRT]

    # Sanity check: Enable all backends except for TensorRT
    # targets = [Target.TVM_GPU_AUTOTVM, Target.CUDNN, Target.CUBLAS]
    # We coudln't figure out how to support CUBLAS in Jetson yet
    # It shouldn't be a big deal though given TensorRT uses CuBLAS internally
    # targets = [Target.TVM_GPU_AUTOTVM, Target.CUDNN, Target.TENSORRT]
    targets = [Target.TVM_GPU_AUTOTVM, Target.CUDNN, Target.TENSORRT, Target.CUBLAS]
    #targets = [Target.TVM_GPU_AUTOTVM, Target.CUDNN]
    #targets = [Target.TVM_GPU_AUTOTVM, Target.TENSORRT]#, Target.CUBLAS]

    batch_size = 1
    backendop_lib = setup_backend_op_lib(relay_expr, targets, batch_size, hw_name)

    # Optimizing graph
    optimizer = CompGraphOptimizer(backendop_lib, targets)

    # visualize_network(relay_expr, "o3_nasnet_test", comp_graph.expr2node)
    """
    Warning(@Soo): Note that current DP optimizer does not work for patterns with more than one root.
    For example, Conv     Conv (Two parallel convolution) case can't be handled
                   \       /
                      ReLU
    Following lines need to be modified to afford more than two roots
    - pat.get_relay_pattern().match(f_expr)

    This is because of inherent limitation of Relay pattern and
    the discrepancy between what Relay pattern supports and how TVM fusion strategy works.
    We can come back to this later if this is critical to performance, which is unlikely for now given networks we have.
    """
    optimized_match = optimizer.optimize(comp_graph, hw_name)

    print("[Op-Level: DP] It finished optimizing comp graph and assigning backend ops to Relay Expr (backend attr)")

    # optimized_match, post_order_match_result = optimizer.get_optimized_match(comp_graph)

    # print("Match result: ", optimized_match)
    # post_order_match_result is for debugging to check if it matches the final result from the TVM DP fusion pass
    # print("Match result")
    # for idx, pair in enumerate(post_order_match_result):
    #     print(idx, pair)
    # # Debug (@soo)
    # print_matching_final(comp_graph, optimizer.loc2match)
    # print("-"*40)

    backendop_lib.save_to_log(hw_name)

    return optimized_match, relay_expr, backendop_lib, n_relay_nodes


@tvm._ffi.register_func("relay.transform.optimizer.run_two_level_opt")
def run_two_level_opt(relay_expr):
    """
    Two-level optimization pass
    First level is for backend operators from TVM and CuDNN, which are not external compilers.
    Second level is for external compilers such as TensorRT.
    We have two separate levels because
    1) First level remove need to enumerate all possible extenral compiler patterns
    2) First level provide us with optimal solutions for backend operators.
    Thus, for each backend operators, second level only needs to consider between
    TensorRT operators and chosen optimal backend operators from first level.

    Parameters
    ----------
    relay_expr : tvm.relay.expr
        Relay IR for computation graph

    Returns
    -------
    matched_relay_expr : tvm.relay.expr
        The result matching between backend operators and Relay operators
    """

    # It is a function if you get it from last pass of Relay build
    print("[Python side] Run two-level optimization")
    # visualize_network(relay_expr, "o3_mobilenet_v2")
    # op-level optimization: DP with all backends but external compilers, e.g., TensorRT
    func_expr = relay_expr
    optimized_match, relay_expr, backendop_lib, n_relay_nodes = run_op_level_opt(relay_expr)
    print("[Python side] Op-level optimization is done")

    # Debug
    # OpMatchLogger().save(relay_expr, optimized_match)
    # print(OpMatchReader().read(relay_expr))

    # subgraph-level optimization for external compilers
    # Translate optimized_match into OpState class that can be used for evolutionary search
    group_id_to_exprs_anno = MatchToOpGroupTranslator().translate(relay_expr, optimized_match)

    # Prepare OpStateToMatchTranslator
    # This translates Op State to optimized_match with the following two dictionaries
    op_state_to_match_translator = OpStateToMatchTranslator(optimized_match, group_id_to_exprs_anno)

    # Run evolutionary search
    n_ops_after_first_level = len(group_id_to_exprs_anno.keys())
    print(f"# of matched operators from first level : {n_ops_after_first_level}")

    # Unnecessary ops include external compiler ops chosen from first level or
    # Ops that shouldn't be assigned to external compiler such as Tuple or TupleGetItem
    n_ops = len(op_state_to_match_translator.state_id_to_group_id)
    print(f"# of matched operators from first level after excluding unnecessary ops: {n_ops}")

    # On the second level, Consider only ops that are not assigned to TensorRT
    # Extract ops that are not assigned to TensorRT

    # Warning(@soo): Network name is hardcoded for now. We can fix it later
    hw_name = func_expr.attrs[HW_FUNC_ATTR]
    net_name = func_expr.attrs[NETWORK_FUNC_ATTR]
    printe(f"Network name: {net_name}")

    # if net_name == "nasneta":
    #     OPT_LEVEL.set(2)

    # Save fisrt layer best results
    first_layer_best_match_log_path = f"{BEST_MATCH_LOG}_{net_name}_{hw_name}_op_level.log"
    OpMatchLogger().save(relay_expr, optimized_match, log_path=first_layer_best_match_log_path)

    # Save it for user-defined fusion pass to measure end-to-end perf
    match_path = f"{LOG_PATH}/user_defined_match_{net_name}_{hw_name}.log"
    OpMatchLogger().save(relay_expr, optimized_match, log_path=match_path)

    # n_ops for each network (it may vary depending on trials)
    # Search space size: 2^n_ops
    # ResNet: 169 -> 65 -> 19
    # ResNext: 169 -> 79
    # Nasrnn: 311 -> 97
    # NasNet: 683 -> 312
    # BERT: 169 -> 96

    # Evolutionary search hyperparameter info
    # Example: pop_size * max_iter (=1) roughly takes 2~4 secs
    # References: ResNet50, 10 * 56 (560) takes 1559.51 s (2.78 secs per pop size per iteration)
    # References: ResNext50, 20 * 100 (2000) takes 4474 s (2.27 secs per pop size per iteration)


    # 100 * 200 (20000) leads to out of memory issues. We attribute this to large population issue of deap lib
    # Note that some of individuals may not be measured in each generation if they are measured anytime earlier
    # visualize_network(relay_expr, "o3_mobilenet_v2_after_match")
    # cx_prob = 0.8, mut_prob = 0.5, resnet50: 2.512
    if n_ops > 0:
        ev_searcher = EvolutionarySearcher(op_state_to_match_translator, relay_expr, net_name, hw_name,
                                           n_ops=n_ops,
                                           # pop_size=10, max_iter=2)  # For simpler debugging
                                           # pop_size=10, max_iter=5) # For debugging
                                           pop_size=50,   max_iter=100000) # For experiment
        second_opt_match = ev_searcher.search(rnd_seed=64)
    else:
        second_opt_match = optimized_match
        printe("No need for subgraph optimization because either 1) op optimization pass only chose Ext compiler ops"
               + " or 2) External compiler can't support ops that are not assigned to external compilers")
    # OpMatchLogger().save(relay_expr, second_opt_match, log_path=USER_DEFINED_MATCH_LOG)
    #second_opt_match = ev_searcher.search_test(rnd_seed=64)

    # print(f"fusion dic (before merge): {optimized_match}")
    # optimized_match = ExtCompilerOpMerger(optimized_match).merge(relay_expr)
    # print(f"fusion dic (after  merge): {optimized_match}")

    # Update backend information to corresponding best match
    second_layer_best_match_log_path = f"{BEST_MATCH_LOG}_{net_name}_{hw_name}.log"
    second_opt_match = OpMatchReader().read(relay_expr, second_layer_best_match_log_path)

    return second_opt_match

@tvm._ffi.register_func("relay.transform.optimizer.run_dp")
def run_dp(relay_expr):
    run_op_level_opt(relay_expr)

# """
# This is still work in progress.
#
# Pros: it gives us upper bound
# Cons: How do you consider all possible TensorRT operators? I don't have good answers to that yet.
#
# """
# @tvm._ffi.register_func("relay.transform.optimizer.run_exhaustive_search")
# def run_exhaustive_search(relay_expr):
#     # It is a function if you get it from last pass of Relay build
#     hw_name = relay_expr.attrs[HW_FUNC_ATTR]
#     print("Optimizing on the Python side")
#     # print("Relay expression")
#     # print(relay_expr)
#     # profile_ops_in_net(relay_expr, "bert", "tensorrt", "rtx2070")
#     # import sys
#     # sys.exit(0)
#     # visualize_network(relay_expr, "o3_bert")
#     relay_expr = get_function_body(relay_expr)
#
#     comp_graph = ComputationGraph(relay_expr)
#
#     # Warning: ResNet-8 doesn't have tuned operators / CuDNN doesn't work for ResNet-8
#     # target_backend = None # Consider all targets
#
#     # Sanity check: AutoTVM
#     # targets = [Target.TVM_GPU_AUTOTVM]
#
#     # Sanity check: Only CuDNN
#     # targets = [Target.TVM_GPU_AUTOTVM, Target.CUDNN]
#
#     # Enable all backends
#     targets = [Target.TVM_GPU_AUTOTVM, Target.CUBLAS, Target.CUDNN, Target.TENSORRT]
#     batch_size = 1
#     backendop_lib = setup_backend_op_lib(relay_expr, targets, batch_size, hw_name)
#
#     # Optimizing graph
#     print("Computation graph created")
#     optimizer = ExhaustiveSearcher(backendop_lib, targets)
#     print("Optimizer created")
#     optimizer.optimize(comp_graph, hw_name)
#     print("It's optimized")
#     optimized_match, post_order_match_result = optimizer.get_optimized_match(comp_graph)
#
#     # print("Match result: ", optimized_match)
#     # post_order_match_result is for debugging to check if it matches the final result from the TVM DP fusion pass
#     print("Match result")
#     for idx, pair in enumerate(post_order_match_result):
#         print(idx, pair)
#     # Debug (@soo)
#     print_matching_final(comp_graph, optimizer.loc2match)
#     print("-" * 40)
#     # print("Optimized match")
#     # print(optimized_match)
#
#     # print(f"fusion dic (before merge): {optimized_match}")
#     # optimized_match = ExtCompilerOpMerger(optimized_match).merge(relay_expr)
#     # print(f"fusion dic (after  merge): {optimized_match}")
#
#     backendop_lib.save_to_log(hw_name)
#
#     return optimized_match

# # Test script with ResNet-8
# if __name__ == "__main__":
#     batch_size = 1
#     num_class = 1000

#     # Resnet-8
# #     image_shape = (3, 28, 28)
# #     mod, params = testing.resnet.get_workload(num_layers=8, batch_size=batch_size, image_shape=image_shape)
# #     relay_expr = mod["main"].body

#     # Chain graph
#     out_channels = 16
#     batch_size = 1

#     data = relay.var("data", relay.TensorType((batch_size, 3, 224, 224), "float32"))
#     conv_weight = relay.var("weight")
#     dense_weight = relay.var("weight")
#     bn_gamma = relay.var("bn_gamma")
#     bn_beta = relay.var("bn_beta")
#     bn_mmean = relay.var("bn_mean")
#     bn_mvar = relay.var("bn_var")

#     simple_net = relay.nn.conv2d(
#         data=data, weight=conv_weight, kernel_size=(3, 3), channels=out_channels, padding=(1, 1)
#     )
# #     simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
# #     simple_net = relay.nn.relu(simple_net)
# #     simple_net = relay.nn.relu(simple_net)
#     relay_expr = simple_net

#     optimize_comp_graph(relay_expr)

# Deprecated get_user_fusion
# fusion_dic[relay_expr] = "0-tvmgpu-autotvm_relu"  # Relu
# fusion_dic[relay_expr.args[0]] = "1-tvmgpu-autotvm_conv2d"  # Conv
# fusion_dic[relay_expr.args[0].args[0]] = "1-tvmgpu-autotvm_conv2d"  # Relu
# fusion_dic[relay_expr.args[0].args[1]] = "1-tvmgpu-autotvm_conv2d"  # Param
# fusion_dic[relay_expr.args[0].args[0].args[0]] = "3-cudnn_conv2d"  #
# fusion_dic[relay_expr.args[0].args[0].args[0].args[1]] = "3-cudnn_conv2d"  # Param
# fusion_dic[relay_expr.args[0].args[0].args[0].args[0]] = "3-cudnn_conv2d"  # Data

# fusion_dic[relay_expr] = "0-tvmgpu-no-tuning_relu" # Relu
# fusion_dic[relay_expr.args[0]] = "1-tvmgpu-no-tuning_conv2d" # Conv
# fusion_dic[relay_expr.args[0].args[0]] = "2-tvmgpu-no-tuning_relu"  # Relu
# fusion_dic[relay_expr.args[0].args[1]] = "1-tvmgpu-no-tuning_conv2d" # Param
# fusion_dic[relay_expr.args[0].args[0].args[0]] = "3-tvmgpu-no-tuning_conv2d"  #
# fusion_dic[relay_expr.args[0].args[0].args[0].args[1]] = "3-tvmgpu-no-tuning_conv2d" # Param
# fusion_dic[relay_expr.args[0].args[0].args[0].args[0]]= "3-tvmgpu-no-tuning_conv2d" # Data

# Enable External compiler merging or not
# print(f"fusion dic (before merge): {fusion_dic}")
# fusion_dic = ExtCompilerOpMerger(fusion_dic).merge(relay_expr)
# print(f"fusion dic (after  merge): {fusion_dic}")
