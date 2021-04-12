import tvm

def get_next_expr_after_match(relay_expr, prev_relay_expr, depth): 
    target_node = []

    if type(relay_expr) == tvm.relay.expr.Var:
        if relay_expr.name_hint == 'data':
            return [(relay_expr, prev_relay_expr)]
        return [(None, prev_relay_expr)]

    if depth == 0:
        return [(relay_expr, prev_relay_expr)]

    # If it is tuple, you should use tuple_value instead of args
    # Caution: depth or depth-1?
    if type(relay_expr) == tvm.relay.expr.TupleGetItem:
        target_node += get_next_expr_after_match(relay_expr.tuple_value, relay_expr, depth-1)
    else:
        for node in relay_expr.args:
            target_node += get_next_expr_after_match(node, relay_expr, depth-1)
#             # FIX: Hacky way to avoid residual connection
#             break

    return target_node

def get_pattern_len(pattern):
    length = 0
    if type(pattern) == tvm.relay.dataflow_pattern.CallPattern:
        for child in pattern.args:
            length = max(length, get_pattern_len(child))
        length +=1
    elif type(pattern) == tvm.relay.dataflow_pattern.TupleGetItemPattern:
        length = get_pattern_len(pattern.tuple)
        length += 1

    return length

def print_matching_final(comp_graph, loc2match):
    idx = -1
    if hash(comp_graph._nodes[idx]) in loc2match:
        graph_str = loc2match[hash(comp_graph._nodes[idx])]["string"]
        reverse_graph_str = ""
        for node_str in graph_str[1:].split('-'):
            reverse_graph_str = node_str + "-" + reverse_graph_str
        print(f"Graph : {reverse_graph_str} (hash: {hash(comp_graph._nodes[idx])})")
        
        tot_cost = loc2match[hash(comp_graph._nodes[idx])]["cost"]
        print(f"Total Cost:{tot_cost}")
        
        print("Matched backend ops (op, cost)")
        for item in loc2match[hash(comp_graph._nodes[idx])]["match"][::-1]:
            op_name, op_cost, _ = item
            print(f"({op_name}, {op_cost:.2g})")
    else:
        raise Exception('Final matching does not exist.')

def print_matching_debug(comp_graph, loc2match):
    for idx in range(len(comp_graph._nodes)):
        if hash(comp_graph._nodes[idx]) in loc2match:
            graph_str = loc2match[hash(comp_graph._nodes[idx])]["string"]
            reverse_graph_str = ""
            for node_str in graph_str[1:].split('-'):
                reverse_graph_str = node_str + "-" + reverse_graph_str
            print(f"Graph : {reverse_graph_str} (hash: {hash(comp_graph._nodes[idx])})")
            
            tot_cost = loc2match[hash(comp_graph._nodes[idx])]["cost"]
            print(f"Total Cost:{tot_cost:.2g}")

            print("Matched backend ops (op, cost)")

            for item in loc2match[hash(comp_graph._nodes[idx])]["match"][::-1]:
                op_name, op_cost = item
                print(f"({op_name}, {op_cost:.2g})")