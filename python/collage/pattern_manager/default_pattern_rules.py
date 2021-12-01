from collage.utils import (
                        is_var_node, 
                        is_constant_node, 
                        is_tuple_node, 
                        is_tuplegetitem_node,
                        get_op_pattern,
                        is_call_node,
                        get_args,
                        is_var,
                    )
from tvm.relay.dataflow_pattern import (
                        is_op, 
                        wildcard, 
                        is_tuple_get_item, 
                        is_tuple, is_constant, 
                        WildcardPattern,
                        CallPattern,
                        ConstantPattern,
                        VarPattern,
                    )
from .base_pattern_rule import BasePatternRule, BasePatternGenerator
from collage.pattern_manager.pattern_language import Pattern

# @ Sung: Check all nodes between src and sink by using fcheck (checks fusion conditions)
# Start from sink
# Gotta be dfs
def check_path(src, node, fcheck, path = [], paths = []):
    path.append(node)
    if src == node:
        assert(len(path))
        paths.append(path.copy())

    elif is_var_node(node) or is_constant_node(node):
        pass
    elif fcheck(node, node==src):
        children = []
        if is_tuple_node(node):
            children = node.fields
        elif is_tuplegetitem_node(node):
            children = [ node.tuple ]
        elif is_call_node(node):
            children = node.args
        else:
            raise Exception(f"Unsupported type ({type(node)})")

        for child in children:
            check_path(src, child, fcheck, path, paths)

    out = path.pop()
    assert(node == out)
    


def generate_relay_pattern_node(node):
    if is_tuple_node(node):
        return is_tuple(), len(node.fields)
    #elif is_tuplegetitem_node(node):
    #    return is_tuple_get_item, 2
    elif is_call_node(node):
        return is_op(node.op.name), len(node.args)
    elif is_constant_node(node):
        return is_constant(), 0
    elif is_var_node(node):
        return is_var(), 0
    else:
        raise Exception(f"Unsupported type ({type(node)})")


# NOTE: Seems like relay pattern matching considers pointer of pattern node.
#       Also, relay pattern string can be misleading
# for example,
#         is_conv2d = is_op('nn.conv2d')(is_var(), is_var())
#         path1 = is_op('nn.relu')(is_conv2d)
#         path2 = is_op('nn.leaky_relu')(is_conv2d)
#         diamond = is_op('add')(path1, path2)
#         --> CallPatternNode(Op(add), [CallPatternNode(Op(nn.relu), [CallPatternNode(Op(nn.conv2d), [VarPattern(),VarPattern()])]), CallPatternNode(Op(nn.leaky_relu), [CallPatternNode(Op(nn.conv2d), [VarPattern(),VarPattern()])])])
#
#         This diamond pattern does not match with the following expr
#            inp1 = relay.var('input1')
#            inp2 = relay.var('input2')
#            weight1 = relay.var('weight1')
#            weight2 = relay.var('weight2')
#            conv2d1 = relay.op.nn.conv2d(inp1, weight1)
#            conv2d2 = relay.op.nn.conv2d(inp2, weight2)
#            relu = relay.op.nn.relu(conv2d1)
#            leaky_relu = relay.op.nn.leaky_relu(conv2d2, alpha=0)
#            out = relu + leaky_relu


def build_pattern_with_map(src, node, nodeToPatternMap):
    if node in nodeToPatternMap:
        rpattern = nodeToPatternMap[node][0]

        children = []
        if is_var_node(node) or is_constant_node(node):
            pass
        elif is_tuple_node(node):
            children = node.fields
        elif is_tuplegetitem_node(node):
            children = [ node.tuple ]
        elif is_call_node(node):
            children = node.args
        else:
            raise Exception(f"Unsupported type ({type(node)})")

        if node == src:
            return rpattern

        operands = [ build_pattern_with_map(src, child, nodeToPatternMap) for child in children ]
        return rpattern(*operands)
    else:
        return wildcard()


def generate_relay_pattern(src, sink, paths = None, cur_pattern_type = None, nodeToPatternMap = dict()):
    if paths is None:
        # Handle single node
        assert(not (is_constant_node(src) and is_var_node(src)))
        assert(cur_pattern_type is None)
        rpattern, num_operands = generate_relay_pattern_node(sink)
        operands = [wildcard() for __ in range(num_operands)]
        return rpattern(*operands), get_op_pattern(sink), 1

    else:
        # Handle multiple nodes
        # Create pattern node for all nodes in paths
        cnt = 0
        nodeToPatternMap = dict()
        for path in paths:
            for node in path:
                if node not in nodeToPatternMap:
                    nodeToPatternMap[node] = generate_relay_pattern_node(node)
                    cnt += 1

                # Create pattern node for const/var
                for child in get_args(node):
                    if is_constant_node(child) or is_var_node(child):
                        if child not in nodeToPatternMap:
                            nodeToPatternMap[child] = generate_relay_pattern_node(child)

        assert(src in nodeToPatternMap)
        pnode, num_operands = nodeToPatternMap[src]
        operands = [wildcard() for __ in range(num_operands)]
        nodeToPatternMap[src] = (pnode(*operands), 0) # it's zero cause we already handled.
        rpattern = build_pattern_with_map(src, sink, nodeToPatternMap)

        return rpattern, cur_pattern_type, cnt


def check_and_generate_pattern(src, sink, fcheck, cur_pattern_type, nodeToPatternMap):
    path, paths = [], []
    
    check_path(src, sink, fcheck, path, paths)
    if len(paths):
        return generate_relay_pattern(src, sink, paths, cur_pattern_type, nodeToPatternMap)
    else:
        return None, None, 0


# @sunggg: default pattern generator. 
# To simulate recent fusion engines, it generates patterns with dom tree.
class DefaultPatternGenerator(BasePatternGenerator):
    def generate(self, post_dom_tree, expr):
        generated_patterns = list()
        if is_constant_node(expr) or is_var_node(expr):
            return generated_patterns # returns empty node

        # Chekc anchor node
        if self.pattern_rule.check(expr):
            if not is_tuple_node(expr):
                anchor_pattern, anchor_type, num_ops = generate_relay_pattern(expr, expr)
                # Verify if it is legitimate
                if self.pattern_rule.verify(anchor_pattern):
                    generated_patterns.append(Pattern(anchor_pattern))
        
            def simulate_fusion(src, sink, cur_type, num_ops, nodeToPatternMap = dict()):
                assert(src is not None)
                if is_tuple_node(sink) and (sink in post_dom_tree):
                    simulate_fusion(src, post_dom_tree[sink], cur_type, num_ops, nodeToPatternMap)

                if self.pattern_rule.check(
                        src = src, 
                        sink = sink, 
                        cur_type = cur_type, 
                        num_ops = num_ops
                    ):
                    
                    # @sunggg: a hacky solution for now to ease implementation.
                    # TODO: Separate check/generation
                    def fcheck(*args):
                        return True
                    
                    fusion_pattern, cur_type, num_ops= check_and_generate_pattern(src, sink, fcheck, cur_type, nodeToPatternMap)
                    #num_ops += cur_num_ops

                    # Append identified pattern
                    if self.pattern_rule.verify(fusion_pattern):
                        generated_patterns.append(Pattern(fusion_pattern))

                    # Go deeper
                    if sink in post_dom_tree:
                        simulate_fusion(src, post_dom_tree[sink], cur_type, num_ops, nodeToPatternMap)
            
            # Run fusion simulation
            if expr in post_dom_tree:
                simulate_fusion(src=expr, sink=post_dom_tree[expr], cur_type=anchor_type, num_ops=num_ops)
            
        return generated_patterns
        

# This pattern rule should be singleton
class TVM_PatternRule(BasePatternRule):
    enum2optype = {0:"kElemWise", 1:"kBroadcast", 2:"kInjective", 3:"kCommReduce", 4:"kOutEWiseFusable", 7:"kTuple", 8:"kOpaque"}
    optype2enum = {"kElemWise":0, "kBroadcast":1, "kInjective":2, "kCommReduce":3, "kOutEWiseFusable":4, "kTuple":7, "kOpaque":8}
    MAX_NUM_OPS = 256

    __instance = None
    @staticmethod
    def destroy():
        TVM_PatternRule.__instance = None

    def __init__(self):
        """ Virtually private constructor. """
        if TVM_PatternRule.__instance != None:
            raise Exception("This class should be a singleton!")
        TVM_PatternRule.__instance = self

    @staticmethod
    def op_rule(expr):
        return get_op_pattern(expr) != TVM_PatternRule.optype2enum["kOpaque"]
    
    @staticmethod
    def fusion_rule(src, sink, cur_type, num_ops):
        enum2optype = TVM_PatternRule.enum2optype
        optype2enum = TVM_PatternRule.optype2enum
        MAX_NUM_OPS  = TVM_PatternRule.MAX_NUM_OPS

        def _check_path(src, node, fcheck):
            if src == node:
                return True
            elif is_var_node(node) or is_constant_node(node):
                return True
            elif fcheck(node, node==src):
                children = []
                if is_tuple_node(node):
                    children = node.fields
                elif is_tuplegetitem_node(node):
                    children = [ node.tuple ]
                elif is_call_node(node):
                    children = node.args
                else:
                    raise Exception(f"Unsupported type ({type(node)})")
                
                chk = True
                for child in children:
                    chk = chk and _check_path(src, child, fcheck)
                return chk

            return False

        if num_ops > MAX_NUM_OPS:
            return False

        sink_type = get_op_pattern(sink)
        if cur_type == optype2enum["kOutEWiseFusable"]:
            def fcheck(node, is_sink):
                return get_op_pattern(node) <= optype2enum["kBroadcast"]

            if sink_type <= optype2enum["kInjective"]:
                return _check_path(src, sink, fcheck)

        elif cur_type <= optype2enum["kBroadcast"]:
            def fcheck(node, is_sink):
                kind = get_op_pattern(node)
                if not is_sink:
                    return kind <= optype2enum["kInjective"]
                else:
                    return kind <= optype2enum["kOutEWiseFusable"]

            if sink_type <= optype2enum["kCommReduce"]:
                return _check_path(src, sink, fcheck)

        elif cur_type == optype2enum["kInjective"] or cur_type == optype2enum["kTuple"]:
            def fcheck(node, is_sink):
                return get_op_pattern(node) <= optype2enum["kInjective"]
            return _check_path(src, sink, fcheck)

        elif cur_type == optype2enum["kCommReduce"] or cur_type == optype2enum["kOpaque"]:
            return False

        else:
            raise Exception(f"Unsupported type ({type(sink)}, {enum2optype[cur_type]}, {src})")
        
        return False
        
        
tvm_pattern_rule = TVM_PatternRule()
tvm_pattern_generator = DefaultPatternGenerator(tvm_pattern_rule)

class TRT_PatternRule(BasePatternRule):
    enum2optype = {0:"kElemWise", 1:"kBroadcast", 2:"kInjective", 3:"kCommReduce", 4:"kOutEWiseFusable", 7:"kTuple", 8:"kOpaque"}
    optype2enum = {"kElemWise":0, "kBroadcast":1, "kInjective":2, "kCommReduce":3, "kOutEWiseFusable":4, "kTuple":7, "kOpaque":8}
    MAX_NUM_OPS = 256
    ops_to_exclude = ["image.resize"]
    __instance = None
    
    @staticmethod
    def destroy():
        TRT_PatternRule.__instance = None

    def __init__(self):
        """ Virtually private constructor. """
        if TRT_PatternRule.__instance != None:
            raise Exception("This class should be a singleton!")
        TRT_PatternRule.__instance = self

    @staticmethod
    def op_rule(expr):
        optype2enum = TRT_PatternRule.optype2enum
        ops_to_exclude = TRT_PatternRule.ops_to_exclude

        return (get_op_pattern(expr) != optype2enum["kOpaque"]) and (expr.op.name not in ops_to_exclude)
    
    # TensorRT seems to follow similar algorithm with TVM.
    # For simplicity, we use the same fusion rule for now.
    # TODO: More accurate TensorRT specification
    @staticmethod
    def fusion_rule(src, sink, cur_type, num_ops):
        enum2optype = TRT_PatternRule.enum2optype
        optype2enum = TRT_PatternRule.optype2enum
        MAX_NUM_OPS  = TRT_PatternRule.MAX_NUM_OPS

        def _check_path(src, node, fcheck):
            if src == node:
                return True
            elif is_var_node(node) or is_constant_node(node):
                return True
            elif fcheck(node, node==src):
                children = []
                if is_tuple_node(node):
                    children = node.fields
                elif is_tuplegetitem_node(node):
                    children = [ node.tuple ]
                elif is_call_node(node):
                    children = node.args
                else:
                    raise Exception(f"Unsupported type ({type(node)})")
                
                chk = True
                for child in children:
                    chk = chk and _check_path(src, child, fcheck)
                return chk

            return False

        if num_ops > MAX_NUM_OPS:
            return False
        
        sink_type = get_op_pattern(sink)
        if cur_type == optype2enum["kOutEWiseFusable"]:
            def fcheck(node, is_sink):
                return get_op_pattern(node) <= optype2enum["kBroadcast"]

            if sink_type <= optype2enum["kInjective"]:
                return _check_path(src, sink, fcheck)

        elif cur_type <= optype2enum["kBroadcast"]:
            def fcheck(node, is_sink):
                kind = get_op_pattern(node)
                if not is_sink:
                    return kind <= optype2enum["kInjective"]
                else:
                    return kind <= optype2enum["kOutEWiseFusable"]

            if sink_type <= optype2enum["kCommReduce"]:
                return _check_path(src, sink, fcheck)

        elif cur_type == optype2enum["kInjective"] or cur_type == optype2enum["kTuple"]:
            def fcheck(node, is_sink):
                return get_op_pattern(node) <= optype2enum["kInjective"]
            return _check_path(src, sink, fcheck)
            
        elif cur_type == optype2enum["kCommReduce"] or cur_type == optype2enum["kOpaque"]:
            return False
        else:
            raise Exception(f"Unsupported type ({type(sink)}, {enum2optype[cur_type]}, {src})")
        
        return False

    @staticmethod
    def verify(pattern):
        q = [ pattern ]
        while len(q):
            cur = q.pop()
            if isinstance(cur, WildcardPattern):
                pass
            elif isinstance(cur, CallPattern):
                if isinstance(cur.op, ConstantPattern) or isinstance(cur.op, VarPattern):
                    pass
                else:
                    op_name = cur.op.expr.name

                    if op_name in TRT_PatternRule.ops_to_exclude:
                        return False
                    q.extend(cur.args)
            elif isinstance(cur, TuplePattern):
                q.extend(cur.fields)
            elif isinstance(cur, TupleGetItemPattern):
                q.append(cur.tuple_value)
            else:
                raise Exception(f"Unexpected expression type, {type(cur)}")

        return True

trt_pattern_rule = TRT_PatternRule() 
trt_pattern_generator = DefaultPatternGenerator(trt_pattern_rule)

"""
def _tvm_pattern_rule(expr, post_dom_tree, verify, backend=None):
    enum2optype = {0:"kElemWise", 1:"kBroadcast", 2:"kInjective", 3:"kCommReduce", 4:"kOutEWiseFusable", 7:"kTuple", 8:"kOpaque"}
    optype2enum = {"kElemWise":0, "kBroadcast":1, "kInjective":2, "kCommReduce":3, "kOutEWiseFusable":4, "kTuple":7, "kOpaque":8}


    def run_fuse(src, sink, cur_pattern_type = None, cur_num_op = 0, nodeToPatternMap = dict()):
        assert(src is not None)
        if cur_pattern_type == optype2enum["kOpaque"] or cur_num_op > MAX_NUM_OPS:
            return None

        if is_tuple_node(sink):
            # Go deeper if possible
            if sink in post_dom_tree:
                run_fuse(src, post_dom_tree[sink], cur_pattern_type, cur_num_op, nodeToPatternMap)
            return None

        sink_type = get_op_pattern(sink)
        # NOTE: This is current assumption. May not be true all the time.
        num_nodes = 0
        cur_relay_pattern = None

        if src == sink:
            # Handle single op
            cur_relay_pattern, cur_pattern_type, num_nodes = generate_relay_pattern(src, sink)
        else:
            if cur_pattern_type == optype2enum["kOutEWiseFusable"]:
                def fcheck(node, is_sink):
                    return get_op_pattern(node) <= optype2enum["kBroadcast"]

                if sink_type <= optype2enum["kInjective"]:
                    cur_relay_pattern, cur_pattern_type, num_nodes = check_and_generate_pattern(src, sink, fcheck, cur_pattern_type, nodeToPatternMap)

            elif cur_pattern_type <= optype2enum["kBroadcast"]:
                def fcheck(node, is_sink):
                    kind = get_op_pattern(node)
                    if not is_sink:
                        return kind <= optype2enum["kInjective"]
                    else:
                        return kind <= optype2enum["kOutEWiseFusable"]

                if sink_type <= optype2enum["kCommReduce"]:
                    cur_relay_pattern, cur_pattern_type, num_nodes = check_and_generate_pattern(src, sink, fcheck, cur_pattern_type, nodeToPatternMap)

            elif cur_pattern_type == optype2enum["kInjective"] or cur_pattern_type == optype2enum["kTuple"]:
                def fcheck(node, is_sink):
                    return get_op_pattern(node) <= optype2enum["kInjective"]
                cur_relay_pattern, cur_pattern_type, num_nodes = check_and_generate_pattern(src, sink, fcheck, cur_pattern_type, nodeToPatternMap)
            elif cur_pattern_type == optype2enum["kCommReduce"] or cur_pattern_type == optype2enum["kOpaque"]:
                pass
            else:
                raise Exception(f"Unsupported type ({type(sink)}, {enum2optype[cur_pattern_type]}, {src})")


        # Invalid pattern
        if num_nodes==0 or cur_relay_pattern is None:
            return None


        cur_num_op += num_nodes

        if verify(cur_relay_pattern):
            self._add_backend_pattern(backend, Pattern(cur_relay_pattern))

        # We may be able to expand
        #if sink in post_dom_tree and src_type != optype2enum["kTuple"]:
        if sink in post_dom_tree:
            run_fuse(src, post_dom_tree[sink], cur_pattern_type, cur_num_op, nodeToPatternMap)

    MAX_NUM_OPS = 256
    if not (is_constant_node(expr) or is_var_node(expr)):
        run_fuse(expr, expr)
"""