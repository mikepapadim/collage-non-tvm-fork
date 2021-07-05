import tvm
from tvm import relay
from tvm.contrib import graph_runtime as runtime
import numpy as np

from .optimizer_utils import get_pattern_len, get_next_expr_after_match
from ..backend_operator.backend_op import get_optimal_backendop
from ..backend_operator.target import *
from .ext_compiler_op_merger import *

try:
    import Queue as Q  # ver. < 3.0
except ImportError:
    import queue as Q

class ExprMatcher:
    def __init__(self, dp_result):
        self._memo = {}
        self._optimized_match = {}
        self._dp_result = dp_result
        self._topo_order_to_op = []

    def match(self, expr):
        self._memo = {}
        self._optimized_match = {}

        dummy_annotation = (9999999, "PYTHON_INVALID_BACKEND_OP")
        self.visit_expr(expr, dummy_annotation)

        return self._optimized_match, self._topo_order_to_op

    # Visit Relay expressions in post-order
    def visit_expr(self, expr, annotation):
        node_type = "INVALID"

        if hash(expr) in self._memo:
           return
        else:
            # memorize this visit to prevent it from visiting twice
            self._memo[hash(expr)] = True

        if hash(expr) in self._dp_result:
            annotation = self._dp_result[hash(expr)]

        # We assume that child class at least have methods for these
        if is_constant_node(expr):
            self.visit_expr_const(expr, annotation)
            node_type = "Const"
        elif is_var_node(expr):
            self.visit_expr_var(expr, annotation)
            node_type = "Var"
        elif is_tuplegetitem_node(expr):
            self.visit_expr_tuplegetitem(expr, annotation)
            node_type = "TupleGetItem"
        elif is_call_node(expr):
            self.visit_expr_call(expr, annotation)
            node_type = expr.op
        elif is_function_node(expr):
            self.visit_expr_func(expr, annotation)
            node_type = "Function"
        elif is_tuple_node(expr):
            self.visit_expr_tuple(expr, annotation)
            node_type = "Tuple"
        else:
            raise Exception(f"Unexpected expression type, {type(expr)}")

        # Add new group id and backend op id to match
        # is_leaf_node = is_constant_node(expr) or is_var_node(expr)
        # if not is_leaf_node:
        if expr not in self._optimized_match:
            # annotation[0] -> group number, annotation[1] -> backend op name
            # Func(Relu2(Conv2(Func(Relu1(Conv1)))))
            # Dictionary
            # Conv1 : 0, tensrort_fused_conv
            # Relu1 : 0, tensrort_fused_conv
            # Conv2 : 1, tensrort_fused_conv
            # Relu2 : 1, tensrort_fused_conv

            # Update backend in the representation
            backend_annotation = create_backend_op_annotation(annotation[0], annotation[1])
            # printe(f"Pair of type and annotation: {backend_annotation}")
            # printe(repr(expr), backend_annotation)
            relay.analysis.update_backend(expr, backend_annotation)

            self._optimized_match[expr] = backend_annotation
            self._topo_order_to_op.append((node_type, self._optimized_match[expr]))
        else:
            raise Exception("Expression should not be visited more than once")

        # print("After each expression", self._optimized_match)

    def visit_expr_const(self, expr, annotation):
        pass

    def visit_expr_var(self, expr, annotation):
        pass

    def visit_expr_tuple(self, expr, annotation):
        for arg in expr.fields:
            self.visit_expr(arg, annotation)

    def visit_expr_tuplegetitem(self, expr, annotation):
        self.visit_expr(expr.tuple_value, annotation)

    def visit_expr_call(self, expr, annotation):
        op, args, attrs, type_args, span = expr.op, expr.args, expr.attrs, expr.type_args, expr.span

        for arg in args:
            self.visit_expr(arg, annotation)

    def visit_expr_func(self, expr, annotation):
        params, body, ret_type, type_params = expr.params, expr.body, expr.ret_type, expr.type_params
        self.visit_expr(body, annotation)

class CompGraphOptimizer:
    def __init__(self, backendop_lib, target_backend=None):
        self._backendop_lib = backendop_lib
        self._target_backend = target_backend

        # Attribute key to pass to N-to-1 lowering pass
        self._bop_attr_key = "backend-op"

        # For printing matched backend ops in ResNet graph
        patterns = self._backendop_lib.get_all_patterns()
        self._pattern_to_name = {}
        for pat in patterns:
            backend_ops = self._backendop_lib.pattern_to_backendops[pat]

            assert len(backend_ops) > 0
            name = backend_ops[0]._op_type.name()
            self._pattern_to_name[pat] = name

        self.loc2match = None
        self._memo = None

        # @Sung: Add driver cost
        self.C = 0.01

        # For Function inputs renaming (recreation)
        # self._local_memo = None
        # self._func_var_id = -1
        # self._has_call = 0

    def optimize(self, comp_graph):
        # HACKY: Reset matched_expr
        comp_graph.reset()
        frontiers = Q.PriorityQueue()
        frontiers.put(comp_graph.get_root())
        pair2match = {}
        self.loc2match = {hash(comp_graph.get_root()): {"match":[], "cost":0, "string":""}}
        while not frontiers.empty():
            # Facilitate the debugging process
            self._backendop_lib.save_to_log()
            f = frontiers.get()
            f_expr = f.get_relay_expr()
            print("="*45)
            if is_call_node(f_expr):
                print(f"(topo_order, op_type) : {f._topological_order}, {f_expr.op}")
            else:
                print(f"(topo_order, op_type) : {f._topological_order}, {type(f_expr)}, Non-call node")

            # print(self._backendop_lib.get_all_patterns())
            for pat in self._backendop_lib.get_all_patterns():
                # print(pat)
                if pat.get_pattern().match(f_expr):
                    # Check if there is an existing frontier with the same goal idx
                    # Conv(Data, Weight)
                    # get_next_expr_after_match -> [Data, Weight]
                    # next_expr_after_match = Conv()
                    assert get_pattern_len(pat.get_pattern()) >= 1
                    # tuple_after_matches = get_next_expr_after_match(f_expr, None, get_pattern_len(pat.get_pattern()))
                    tuple_after_matches = get_next_expr_after_match(f_expr, None, pat.get_pattern())
                    print("The following pattern is matched:", pat.get_pattern())
                    # Consdier only valid nodes
                    tuple_after_matches = [tup for tup in tuple_after_matches if hash(tup[0]) in comp_graph.expr2node]
                    for t_idx, (expr_after_match, prev_expr_after_match) in enumerate(tuple_after_matches):
                        # Get new frontier, matched backend ops, and their costs
                        new_loc = comp_graph.expr2node[hash(expr_after_match)]
                        pat_op, pat_cost = get_optimal_backendop(self._backendop_lib, f_expr, pat, self._target_backend)

                        # Skip update if there is no backend op available for matched pattern
                        if pat_op == None:
                            continue
                        # new_match = self.loc2match[hash(f)]["match"] + [(pat_op, pat_cost, hash(f_expr))]
                        # new_cost = self.loc2match[hash(f)]["cost"] + pat_cost
                        # new_string = self.loc2match[hash(f)]['string'] + "-" + self._pattern_to_name[pat]

                        # Flush matchings from second branch if there are more than one branches
                        if t_idx == 0:
                            new_match = self.loc2match[hash(f)]["match"] + [(pat_op, pat_cost, hash(f_expr))]
                            # @Sung: Add driver cost
                            new_cost = self.loc2match[hash(f)]["cost"] + pat_cost + self.C
                            new_string = self.loc2match[hash(f)]['string'] + "-" + self._pattern_to_name[pat]
                            # print(f"Assign matched op : {pat_op}")
                        else:
                            new_match, new_cost, new_string = [], 0, "+"

                        # Maintain pair2match for keeping track of match results for each branch
                        new_loc.matched_expr[hash(prev_expr_after_match)] = 1
                        out_key = hash(new_loc) # new_loc is node after match
                        in_key = hash(prev_expr_after_match)

                        if out_key not in pair2match:
                            pair2match[out_key] = {}

                        if in_key not in pair2match[out_key] or pair2match[out_key][in_key]["cost"] > new_cost:
                            pair2match[out_key][in_key] = {"match":new_match, "cost":new_cost, "string":new_string}

                        # Update loc2match for final outcome
                        if new_loc.get_n_parents() == new_loc.get_n_matched():
                            if hash(new_loc) not in self.loc2match:
                                frontiers.put(new_loc)

                            new_match, new_cost, new_string = [], 0, ""
                            for _, match_dic in pair2match[out_key].items():
                                new_match += match_dic["match"]
                                new_cost += match_dic["cost"]
                                new_string += match_dic["string"]

                            # Debug logs to compare costs between fused op vs a combination of single ops
                            # if hash(new_loc) in self.loc2match:
                            #     old_cost = self.loc2match[hash(new_loc)]["cost"]
                            #     old_str = self.loc2match[hash(new_loc)]["string"]
                            #     eprint(f"old : ({old_cost:.4f}, {old_str})")
                            #     eprint(f"new : ({new_cost:.4f}, {new_string})")
                            #     eprint("--" * 10)
                            if hash(new_loc) not in self.loc2match or self.loc2match[hash(new_loc)]["cost"] > new_cost:
                                self.loc2match[hash(new_loc)] = {"match": new_match, "cost":new_cost, "string":new_string}

    def get_optimized_match(self, comp_graph):
        assert self.loc2match is not None

        # Get final match (expr, backend_op_name)
        result_idx = -1
        final_match = {}
        fused_group_id = 0
        # print(self.loc2match)
        # print([hash(node) for node in comp_graph._nodes])
        # print(comp_graph._nodes[-1].get_relay_expr())
        for (pat_op, pat_cost, hash_expr) in self.loc2match[hash(comp_graph._nodes[result_idx])]["match"]:
            final_match[hash_expr] = (fused_group_id, pat_op)
            fused_group_id += 1

        optimized_match, post_order_match_result = ExprMatcher(final_match).match(comp_graph.get_relay_expr())

        return optimized_match, post_order_match_result

"""
FrontierGraph and FrontierNode is to allow effective search over matched backend op assignments.
We use DFS to explore all possible combinations of backend ops.
Given the maximum width of graph, DFS is more memory-efficient than BFS.
"""
class FrontierNode:
    def __init__(self, expr, backend_ops):
        self.children = []
        # Relay_expr is before matched backend_ops
        self.relay_expr = expr
        self.backend_ops = backend_ops

    def add_child(self, child):
        self.children.append(child)

    def get_n_children(self):
        return len(self.children)

class FrontierGraph:
    def __init__(self, root):
        self.root = root

"""
We have two separate passes:
1) First pass is for generating backend op assignment tree by matching every possible backend op over graph.
We call this tree as FrontierGraph and it allows exhaustive search over all possible backend op assignments.
2) Second pass is for evaluating all possible backend op assignments for a graph. Note that we only compile an
entire graph rather than compiling multiple subgraphs to evaluate entire graph in DP.
"""
class ExhaustiveSearcher:
    def __init__(self, backendop_lib, target_backend=None):
        self._backendop_lib = backendop_lib
        self._target_backend = target_backend

        # Attribute key to pass to N-to-1 lowering pass
        self._bop_attr_key = "backend-op"

        # For printing matched backend ops in ResNet graph
        patterns = self._backendop_lib.get_all_patterns()
        self._pattern_to_name = {}
        for pat in patterns:
            backend_ops = self._backendop_lib.pattern_to_backendops[pat]

            assert len(backend_ops) > 0
            name = backend_ops[0]._op_type.name()
            self._pattern_to_name[pat] = name

        self.memo_map = {}

        # Key: Expr / Value: backend op id + name
        self._backend_op_id = 0
        self.frontier_graph = None

    def debug_print(self, f, f_expr):
        print("=" * 45)
        if is_call_node(f_expr):
            print(f"(topo_order, op_type) : {f._topological_order}, {f_expr.op}")
        else:
            print(f"(topo_order, op_type) : {f._topological_order}, {type(f_expr)}, Non-call node")

    def create_backend_op_annotation(self, backend_op):
        return f"{self._backend_op_id}-{backend_op}"

    def get_backend_ops(self, pattern):
        backend_ops = self._backendop_lib.get_backendops(pattern)
        backend_op_annotations = []
        for op in backend_ops:
            if op.get_target() in self._target_backend:
                backend_op_annotations.append(self.create_backend_op_annotation(op))

        return backend_op_annotations

    def optimize(self, comp_graph):
        # HACKY: Reset matched_expr
        comp_graph.reset()

        # Note that we have Node object (not Relay Expr) in the frontiers
        frontiers = Q.PriorityQueue()
        frontiers.put(comp_graph.get_root())
        self.frontier_graph = FrontierGraph(FrontierNode(comp_graph.get_root().get_relay_expr(), None))
        pair2match = {}

        while not frontiers.empty():
            # Facilitate the debugging process
            self._backendop_lib.save_to_log()

            f = frontiers.get()
            f_expr = f.get_relay_expr()
            f_node = FrontierNode(comp_graph.get_root().get_relay_expr(), None)

            # Debug printing
            self.debug_print(f, f_expr)

            # print(self._backendop_lib.get_all_patterns())
            for pat in self._backendop_lib.get_all_patterns():
                # print(pat)
                if pat.get_pattern().match(f_expr):
                    # Check if there is an existing frontier with the same goal idx
                    # Conv(Data, Weight)
                    # get_next_expr_after_match -> [Data, Weight]
                    # prev_expr_after_match = Conv()
                    assert get_pattern_len(pat.get_pattern()) >= 1

                    tuple_after_matches = get_next_expr_after_match(f_expr, None, pat.get_pattern())
                    print("The following pattern is matched:", pat.get_pattern())

                    # Consdier only valid nodes
                    tuple_after_matches = [tup for tup in tuple_after_matches if hash(tup[0]) in comp_graph.expr2node]
                    for t_idx, (expr_after_match, prev_expr_after_match) in enumerate(tuple_after_matches):
                        # Get new frontier, matched backend ops, and their costs
                        new_loc = comp_graph.expr2node[hash(expr_after_match)]
                        backend_ops = self.get_backend_ops(pat)

                        # Skip update if there is no backend op available for matched pattern
                        if len(backend_ops) == 0:
                            continue

                        # Flush matchings from second branch if there are more than one branches
                        if t_idx == 0:
                            pass
                            # new_string = self.loc2match[hash(f)]['string'] + "-" + self._pattern_to_name[pat]
                            # print(f"Assign matched op : {pat_op}")
                        else:
                            new_match, new_cost, new_string = [], 0, "+"

                        # Maintain pair2match for keeping track of match results for each branch
                        new_loc.matched_expr[hash(prev_expr_after_match)] = 1
                        # new_loc is node after match
                        out_key, in_key = hash(new_loc), hash(prev_expr_after_match)

                        if out_key not in pair2match:
                            pair2match[out_key] = {}

                        if in_key not in pair2match[out_key] or pair2match[out_key][in_key]["cost"] > new_cost:
                            pair2match[out_key][in_key] = {"match": new_match, "cost": new_cost, "string": new_string}

                        # Update loc2match for final outcome
                        if new_loc.get_n_parents() == new_loc.get_n_matched():
                            if hash(new_loc) not in self.loc2match:
                                frontiers.put(new_loc)

                            new_match, new_cost, new_string = [], 0, ""
                            for _, match_dic in pair2match[out_key].items():
                                new_match += match_dic["match"]
                                new_cost += match_dic["cost"]
                                new_string += match_dic["string"]

                            if hash(new_loc) not in self.loc2match or self.loc2match[hash(new_loc)]["cost"] > new_cost:
                                self.loc2match[hash(new_loc)] = {"match": new_match, "cost": new_cost,
                                                                 "string": new_string}

    def get_optimized_match(self, comp_graph):
        assert self.loc2match is not None

        # Get final match (expr, backend_op_name)
        result_idx = -1
        final_match = {}
        fused_group_id = 0
        # print(self.loc2match)
        # print([hash(node) for node in comp_graph._nodes])
        # print(comp_graph._nodes[-1].get_relay_expr())
        for (pat_op, pat_cost, hash_expr) in self.loc2match[hash(comp_graph._nodes[result_idx])]["match"]:
            final_match[hash_expr] = (fused_group_id, pat_op)
            fused_group_id += 1

        optimized_match, post_order_match_result = ExprMatcher(final_match).match(comp_graph.get_relay_expr())

        return optimized_match, post_order_match_result




    # # pylint: disable=no-else-return
    # def visit(self, expr):
    #     """Apply the visitor to an expression."""
    #     if expr in self.memo_map:
    #         return self.memo_map[expr]
    #
    #     if isinstance(expr, relay.Function):
    #         res = self.visit_function(expr)
    #     elif isinstance(expr, relay.Call):
    #         res = self.visit_call(expr)
    #     elif isinstance(expr, relay.Var):
    #         res = self.visit_var(expr)
    #     elif isinstance(expr, relay.Tuple):
    #         res = self.visit_tuple(expr)
    #     elif isinstance(expr, relay.TupleGetItem):
    #         res = self.visit_tuple_getitem(expr)
    #     elif isinstance(expr, relay.Constant):
    #         res = self.visit_constant(expr)
    #     else:
    #         raise Exception("warning unhandled case: {0}".format(type(expr)))
    #
    #     self.memo_map[expr] = res
    #
    #     return res
    #
    # def visit_function(self, _):
    #     raise NotImplementedError()
    #
    # def visit_call(self, _):
    #     raise NotImplementedError()
    #
    # def visit_var(self, _):
    #     raise NotImplementedError()
    #
    # def visit_tuple(self, _):
    #     raise NotImplementedError()
    #
    # def visit_tuple_getitem(self, _):
    #     raise NotImplementedError()
    #
    # def visit_constant(self, _):
    #     raise NotImplementedError()
    #
