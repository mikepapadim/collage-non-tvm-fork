from tvm.relay.dataflow_pattern import *

def find_depth(relay_pattern):
    if relay_pattern is None:
        return 0
    children = []

    if isinstance(relay_pattern, TuplePattern):
        children = relay_pattern.fields
    elif isinstance(relay_pattern, TupleGetItemPattern):
        children = [ relay_pattern.tuple ]
    elif isinstance(relay_pattern, CallPattern):
        children = relay_pattern.args
    elif isinstance(relay_pattern, WildcardPattern) or isinstance(relay_pattern, VarPattern) or isinstance(relay_pattern, ConstantPattern):
        return 0
    else:
        raise Exception(f"{type(relay_pattern)} is not handled yet.")

    depth = 0
    if children is not None:
        for child in children:
            depth = max(depth, find_depth(child))
    return depth+1


# currently the Pattern class does not add any additional attributes from TVM's dataflow patterns
class Pattern(object):
    def __init__(self, relay_pattern):
        self._relay_pattern = relay_pattern
        self._name = str(relay_pattern)
        self._depth = find_depth(relay_pattern)

    def __eq__(self, another):
        return isinstance(another, Pattern) and self._relay_pattern == another._relay_pattern

    def __hash__(self):
        return hash(self._name)

    def __repr__(self):
        return self._name

    def match(self, expr):
        return self._relay_pattern.match(expr)

    def set_name(self, op_type):
        self._name = op_type

    def get_name(self):
        return self._name

    def get_relay_pattern(self):
        return self._relay_pattern

    def get_depth(self):
        return self._depth

#     def get_relay_pattern_tree(self):
#         return self._relay_pattern_tree
