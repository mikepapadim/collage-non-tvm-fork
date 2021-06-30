import tvm
from tvm import relay

def _traverse_expr(node, node_dict):
    if node in node_dict:
        return
    # if isinstance(node, relay.op.op.Op):
    #    return
    if isinstance(node, tvm.ir.op.Op):
        return

    # print("{} : {}".format(node, type(node)))
    node_dict[node] = len(node_dict)
    print(node.backend)

data = relay.var("data", shape=(10, 10))
expr = relay.nn.relu(data)
relay.analysis.update_backend(expr, "wow")
relay.analysis.update_backend(data, "wow2")
node_dict = {}
relay.analysis.post_order_visit(expr, lambda node: _traverse_expr(node, node_dict))