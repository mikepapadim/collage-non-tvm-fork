import tvm
import tvm.relay as relay
import tvm.relay.testing as testing
from graphviz import Digraph

# from ..workloads.onnx_workloads import get_network_from_onnx
from ..workloads.torch_workloads import get_network_from_torch
import argparse

def args_checker(args, parser):
    is_missing_arg = not args.network
    # is_missing_arg |= not args.target

    if is_missing_arg:
        parser.error('Make sure you input all arguments')

def get_args():
    parser = argparse.ArgumentParser()
    # Default type is string for argparse
    parser.add_argument("-n", "--network", help="name of a neural network")
    args = parser.parse_args()

    args_checker(args, parser)
    return args

def get_resnet_8():
    batch_size = 1
    image_shape = (3, 28, 28)
    n_layers = 8

    mod, _ = testing.resnet.get_workload(
        num_layers=n_layers, batch_size=batch_size, image_shape=image_shape)

    return mod

def _traverse_expr(node, node_dict):
    if node in node_dict:
        return
    # if isinstance(node, relay.op.op.Op):
    #    return
    if isinstance(node, tvm.ir.op.Op):
        return

    # print("{} : {}".format(node, type(node)))
    node_dict[node] = len(node_dict)

def visualize_network(expr, network_name):
    node_color = 'greenyellow'

    dot = Digraph(format='pdf')
    dot.attr(rankdir='BT')
    # dot.attr('node', shape='box')

    node_dict = {}
    relay.analysis.post_order_visit(expr, lambda node: _traverse_expr(node, node_dict))

    for node, node_idx in node_dict.items():

        if isinstance(node, relay.Function):
            # elif isinstance(node, relay.expr.Function):
            print(f'node_idx: {node_idx}, Function(body={node_dict[node.body]})')
            dot.node(str(node_idx), f'Function ({node_idx})', shape='doubleoctagon')
            dot.edge(str(node_dict[node.body]), str(node_idx))

        elif isinstance(node, relay.expr.Var):
            print(
                f'node_idx: {node_idx}, Var(name={node.name_hint}, type=Tensor[{tuple(node.type_annotation.shape)}, {node.type_annotation.dtype}])')
            dot.node(str(node_idx), \
                     f'{node.name_hint} ({node_idx}):\nTensor[{tuple(node.type_annotation.shape)}, {node.type_annotation.dtype}]', \
                     shape='rectangle'
                     )

        elif isinstance(node, relay.Constant):
            print(f'node_idx: {node_idx}, Constant(type=Tensor[{tuple(node.data.shape)}, {node.data.dtype}])')
            dot.node(str(node_idx), \
                     f'Constant ({node_idx}):\nTensor[{tuple(node.data.shape)}, {node.data.dtype}]', \
                     shape='rectangle'
                     )

        elif isinstance(node, relay.expr.Call):
            args = [node_dict[arg] for arg in node.args]
            # print(f'node_idx: {node_idx}, Call(op_name={node.op.name}, args={args})')
            # dot.node(str(node_idx), f'Call(op={node.op.name})')
            if isinstance(node.op, tvm.relay.Function):
                print(f'node_idx: {node_idx}, Call(Function({node_dict[node.op.body]}))')
                dot.node(str(node_idx), f'Call ({node_idx})(Function({node_dict[node.op.body]}))', shape='ellipse',
                         style='filled', color=node_color)
            else:
                print(f'node_idx: {node_idx}, Call(op_name={node.op.name}, args={args})')
                dot.node(str(node_idx), f'Call ({node_idx})(op={node.op.name})', shape='ellipse', style='filled', color=node_color)
            for arg in args:
                dot.edge(str(arg), str(node_idx))
        elif isinstance(node, relay.expr.TupleGetItem):
            print(f'node_idx: {node_idx}, TupleGetItem(tuple={node_dict[node.tuple_value]}, idx={node.index})')
            dot.node(str(node_idx), f'TupleGetItem ({node_idx})(idx={node.index})', shape='ellipse', style='filled', color=node_color)
            dot.edge(str(node_dict[node.tuple_value]), str(node_idx))
        elif isinstance(node, relay.expr.Tuple):
            args = [node_dict[field] for field in node.fields]
            print(f'node_idx: {node_idx}, Tuple(fields=none)')
            dot.node(str(node_idx), f'Tuple ({node_idx})(fileds=none)', shape='ellipse', style='filled', color=node_color)
            for arg in args:
                dot.edge(str(arg), str(node_idx))
        else:
            raise RuntimeError(f'Unknown node type. node_idx: {node_idx}, node: {type(node)}')

    dot.render(f'analysis/results/net_figs/{network_name}.gv')

if __name__ == "__main__":
    args = get_args()
    print(args)

    # mod = get_resnet_8()
    # mod, _, _, _ = get_network_from_onnx(args.network, batch_size=1)
    mod, _, _, _ = get_network_from_torch(args.network, batch_size=1)
    visualize_network(mod["main"], args.network)






