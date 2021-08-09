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

# TODO: handle conv2d in the diamond edge
def name_relay_pattern(pattern, idMap = None, cnt = 0):
    if idMap is None:
        idMap = dict()



    goDeeper = True
    if isinstance(pattern, WildcardPattern):
        node_str = ""
    else:
        if (pattern not in idMap):
            idMap[pattern] = cnt
            cnt += 1

        uid = idMap[pattern]
        node_str = f"{uid}-"

    if isinstance(pattern, TuplePattern):
        node_str += "Tuple"
        children = pattern.fields
    elif isinstance(pattern, TupleGetItemPattern):
        node_str += "TupleGetItem"
        children = [ pattern.tuple ]
    elif isinstance(pattern, WildcardPattern):
        node_str += "*"
        goDeeper = False
        children = []
    elif isinstance(pattern, VarPattern) or isinstance(pattern, ConstantPattern):
        node_str += "Var/Const"
        goDeeper = False
        children = []
    elif isinstance(pattern, CallPattern):
        if isinstance(pattern.op, VarPattern) or isinstance(pattern.op, ConstantPattern):
            node_str += "Var/Const"
            goDeeper = False
            children = []
        else:
            node_str += str(pattern.op)
            children = pattern.args

    else:
        raise Exception(f"{type(relay_pattern)} is not handled yet.")

    if goDeeper:
        name = node_str + "["
        for i, child in enumerate(children):
            if i > 0:
                name += ", "
            _name, _cnt = name_relay_pattern(child, idMap, cnt)
            name += _name
            cnt = _cnt

        name += "]"
    else:
        name = node_str

    return name, cnt

# currently the Pattern class does not add any additional attributes from TVM's dataflow patterns
class Pattern(object):
    def __init__(self, relay_pattern, name = None):
        self._relay_pattern = relay_pattern
        #self._name = str(relay_pattern)
        self._name = name_relay_pattern(relay_pattern)[0]
        self._depth = find_depth(relay_pattern)

    def __eq__(self, another):
        return isinstance(another, Pattern) and self._relay_pattern == another._relay_pattern

    def __hash__(self):
        return hash(self._name)

    def __repr__(self):
        return self._name

    def match(self, expr):
        return self._relay_pattern.match(expr)

    #def set_name(self, op_type):
    #    self._name = op_type

    def get_name(self):
        return self._name

    def get_relay_pattern(self):
        return self._relay_pattern

    def get_depth(self):
        return self._depth

#     def get_relay_pattern_tree(self):
#         return self._relay_pattern_tree
