import argparse
from collections import defaultdict
import pandas as pd

from ..backend_operator.utils import *
from ..backend_operator.backend_op_lib import BackendOpLib
from ..backend_operator.target import Target
from ..backend_operator.backend_op import get_optimal_backendop
from .plot_utils import set_plt_font_size
import os
import matplotlib.pyplot as plt

STR_TO_TARGET = {
    "autotvm": Target.TVM_GPU_AUTOTVM,
    "tensorrt": Target.TENSORRT,
}

def args_checker(args, parser):
    is_missing_arg = not args.network
    is_missing_arg |= not args.target
    # is_missing_arg |= not args.dtype
    # is_missing_arg |= not args.batch_size

    if is_missing_arg:
        parser.error('Make sure you input all arguments')

def get_args():
    parser = argparse.ArgumentParser()
    # Default type is string for argparse
    parser.add_argument("-n", "--network", help="name of a neural network")
    parser.add_argument("-t", "--target", help="target backend")
    # parser.add_argument("-dt", "--dtype", help="data type")
    # parser.add_argument("-bs", "--batch-size", type=int, help="batch size")
    args = parser.parse_args()

    args_checker(args, parser)
    return args

def _get_op_name(expr):
    op_name = str(expr.op.name).split("nn.")[-1]
    return op_name

def _measure_single_op(expr, op_perf_dic, target, hw_name):
    is_skip_node = isinstance(expr, tvm.ir.op.Op) or isinstance(expr, relay.Function)
    is_skip_node |= is_tuple_node(expr)
    if is_skip_node:
        return

    is_matched_once = False
    backendop_lib = BackendOpLib.get(hw_name)
    for pat in backendop_lib.get_all_patterns():
        if pat.get_op_type().depth() == 1 and pat.match(expr):
            # Measure an op perf for a given backend
            _, op_cost = get_optimal_backendop(backendop_lib, expr, pat, [target], hw_name)
            op_perf_dic[_get_op_name(expr)] += op_cost

            if is_matched_once:
                raise Exception(f"Following pattern should match only once: {pat.get_pattern()}")
            is_matched_once = True
            # break

    # op_perf_dic["n_nodes"] += 1

    if not is_matched_once and not (is_constant_node(expr) or is_var_node(expr)):
        raise Exception(f"The following expr never matched any pattern: {expr}")

def draw_pie_plot(df, network, col_name, target_name, fig_name):
    fw, fh = 10, 7
    set_plt_font_size()
    df.plot.pie(subplots=True, figsize=(fw, fh), autopct='%1.1f%%', legend=False, fontsize=22,
                title=target_name)

    # Beautify the plot

    # Save figures
    this_code_path = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(f'{this_code_path}/../../analysis/results/plots/{fig_name}')
                #bbox_inches='tight')
    print(f"The following plot is generated: {fig_name}")

def draw_bar_plot(df, network, col_name, target_name, fig_name):
    fw, fh = 12, 7
    set_plt_font_size()
    df[col_name] = df[col_name] / df[col_name].sum() * 100.0
    ax = df.plot.barh(figsize=(fw, fh), legend=True, fontsize=22)

    # Beautify the plot
    plt.title(target_name, fontsize=28)
    # Show numbers on top of the bar
    for p in ax.patches:
        ax.annotate("%.2f" % p.get_width(), (p.get_x() + p.get_width(), p.get_y()),
                    xytext=(5, 10), textcoords='offset points')

    # Save figures
    this_code_path = os.path.dirname(os.path.abspath(__file__))
    plt.tight_layout()
    plt.savefig(f'{this_code_path}/../../analysis/results/plots/{fig_name}')
                #bbox_inches='tight')
    print(f"The following plot is generated: {fig_name}")

def profile_ops_in_net(expr, network, target_name, hw_name):

    # Measure op perf by traversing the network
    op_perf_dic = defaultdict(int)
    target = STR_TO_TARGET[target_name]
    relay.analysis.post_order_visit(expr, lambda node: _measure_single_op(node, op_perf_dic, target, hw_name))

    # Plot op perfs in the pie chart
    # Warning(@Soo): make sure we have dict instead of defaultdict to create plots
    col_name = 'time percentage'
    df = pd.DataFrame.from_dict(op_perf_dic, orient="index", columns=[col_name])

    draw_bar_plot(df, network, col_name, target_name, f"profile_op_rtx_{target_name}_{network}.png")
    # if network == 'bert':
    # else:
        # draw_pie_plot(df, network, col_name,target_name, f"profile_op_rtx_{target_name}_{network}.png")


# if __name__ == "__main__":
#     args = get_args()
#     mod, params, _, _ = get_network_from_torch(args.network, 1)
#     target = STR_TO_TARGET[args.target]
#
#     # Measure op perf by traversing the network
#     # op_perf_dic = {"n_nodes":0} # for sanity check
#     op_perf_dic = defaultdict(list)
#     relay.analysis.post_order_visit(mod["main"], lambda node: _measure_single_op(node, op_perf_dic, target))
#     print(op_perf_dic)
#     # n_nodes = op_perf_dic["n_nodes"]
#     # print(f"# of nodes : {n_nodes}")
#
#     # Plot op perfs in the pie chart
