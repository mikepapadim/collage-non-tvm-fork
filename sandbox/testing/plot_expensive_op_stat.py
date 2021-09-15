from workloads.torch_workloads import get_network_from_torch
import tvm.relay as relay
import tvm
from tvm.relay.transform.backend_operator.utils import *
from tvm.relay.transform.backend_operator.op_config import Config
from tvm.relay.transform.backend_operator.backend_op import extract_subgraph
from collections import Counter
from e2e_perf_logger import EXP_RESULT_PATH
import pandas as pd
import json

EXPENSIVE_OPS = ['nn.conv2d', 'nn.conv3d', 'nn.dense', 'nn.batch_matmul']
EXPENSIVE_OP_TO_PATTERN = {
    'nn.conv2d': Pattern(is_op("nn.conv2d")(wildcard(), wildcard())),
    'nn.conv3d': Pattern(is_op("nn.conv3d")(wildcard(), wildcard())),
    'nn.dense': Pattern(is_op("nn.dense")(wildcard(), wildcard())),
    'nn.batch_matmul': Pattern(is_op("nn.batch_matmul")(wildcard(), wildcard())),
}

NETWORKS = ['bert', 'resnet50', "resnext50_32x4d", "nasneta", "nasrnn", "resnet50_3d", "mobilenet_v2", 'dcgan']

def extract_expensive_op_info(node, op_info_dic, op_type_info_dic, memo, memo_type):
    # Prevent it from vising same node more than once
    if node in memo:
        return
    else:
        memo[node] = True

    # Update expensive operator stats
    if is_call_node(node) and str(node.op) in EXPENSIVE_OPS:
        op_name = str(node.op)
        op_info_dic[op_name] += 1

        # Update op_type_info_dic
        # Extract op configuration
        extracted_node = extract_subgraph(node, EXPENSIVE_OP_TO_PATTERN[op_name])
        op_config = Config(op_name, op_name, extracted_node)
        # print(op_config)

        if op_config not in memo_type:
            op_type_info_dic[op_name] += 1
            memo_type[op_config] = True


if __name__ == "__main__":
    log_path = f"{EXP_RESULT_PATH}/expensive_op_stats.csv"

    net_op_info_dic = {}

    with open(log_path, 'w') as f:
        for net in NETWORKS:
            # Load network
            mod, params, shape_dict, _ = get_network_from_torch(net, batch_size=1)
            mod = tvm.relay.transform.InferType()(mod)

            # Visit each node of networks to profile op information
            op_info_dic, op_type_info_dic, memo, memo_type = Counter(), Counter(), {}, {}
            relay.analysis.post_order_visit(mod["main"],
                                            lambda node: extract_expensive_op_info(node, op_info_dic, op_type_info_dic,
                                                                                   memo, memo_type))
            net_op_info_dic[net] = op_info_dic
            op_info = f"number of expensive ops: {net}, {dict(op_info_dic)}"
            op_type_info = f"number of expensive op types: {net}, {dict(op_type_info_dic)}"
            f.write(op_info+"\n")
            f.write(op_type_info+"\n")
            print(op_info)
            print(op_type_info)

    # Dump this info into a json
    # json_object = json.dumps(net_op_info_dic, indent=2)
    # with open(log_path, 'w') as f:
    #     json.dump(json_object, f)

    # Dump this info into a df
    # df = pd.DataFrame.from_dict(net_op_info_dic, orient="index", dtype='int32')
    # df = df.fillna(0)
    # df.index.name = "network"
    # df.to_json(log_path, orient='index')
    # df.to_csv(log_path)
    # print(df)

