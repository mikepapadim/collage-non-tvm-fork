import tvm
from tvm import relay
import tvm.contrib.graph_executor as runtime
import numpy as np

from .optimizer_utils import *
from ..backend_operator.target import *
from .ordered_pattern_matcher import OrderedPatternMatcher
from .dp_table import *

class CompGraphOptimizer:
    def __init__(self, backendop_lib, target_backend=None):
        self._backendop_lib = backendop_lib
        self._target_backend = target_backend
        self._ordered_pattern_matcher = OrderedPatternMatcher()
        # Attribute key to pass to N-to-1 lowering pass
        self._bop_attr_key = "backend-op"

        # For printing matched backend ops in ResNet graph
        #patterns = self._backendop_lib.get_all_patterns()
        #self._pattern_to_name = {}
        #for pat in patterns:
        #    backend_ops = self._backendop_lib.pattern_to_backendops[pat]

       #     assert len(backend_ops) > 0
            #name = backend_ops[0]._op_name
            #TODO: Check with Soo
       #     self._pattern_to_name[pat] = pat.get_name()

        self.loc2match = None
        self._memo = None

        # @Sung: Add driver cost
        # self.C = 0.01

        # For Function inputs renaming (recreation)
        # self._local_memo = None
        # self._func_var_id = -1
        # self._has_call = 0

    def optimize(self, comp_graph, hw_name):
        # HACKY: Reset matched_expr
        comp_graph.reset()

        frontier_queue = FrontierQueue()
        frontier_queue.put(comp_graph.get_root())

        extractor = MatchInfoExtractor(comp_graph)
        dp_table = DPTable(self._backendop_lib, self._target_backend, hw_name, comp_graph)

        pair2match = {}

        root_expr = comp_graph.get_root().get_relay_expr()
        # dom_tree: <class 'tvm.ir.container.Map'> --> dictionary in Python
        dom_tree = relay.analysis.construct_dom_tree(root_expr, post_dom = True)


        # @Sung: run all pattern generators
        all_exprs = []
        def _traverse_expr(node, node_list):
            if not is_call_node(node):
                return
            if node in node_list:
                return
            if isinstance(node, tvm.ir.op.Op):
                return
            node_list.append(node)

        relay.analysis.post_order_visit(root_expr, lambda expr: _traverse_expr(expr, all_exprs))

        for expr in all_exprs:
            #if expr.op.name == "nn.dense":
            for generator in self._backendop_lib.get_all_pattern_generators():
                 generator.run(dom_tree, expr)


        for pat in self._backendop_lib.get_all_patterns():
            print("Checking... ", pat)


        # for node, dom in dom_tree.items():
        #    print(f"{repr(node)} --> {repr(dom)}\n")
        #    print("\n")

        # Debug
        # hash_to_op = {}
        # for node in comp_graph._nodes:
        #     if not is_var_node(node.get_relay_expr()):
        #         hash_to_op[hash(node)] = node.get_relay_expr().op
        #     else:
        #         hash_to_op[hash(node)] = "var"

        # self.loc2match = {hash(comp_graph.get_root()): {"match":[], "cost":0, "string":""}}
        while not frontier_queue.empty():
            # Facilitate the debugging process
            self._backendop_lib.save_to_log(hw_name)
            f = frontier_queue.get()
            f_expr = f.get_relay_expr()

            print("="*45)
            if is_call_node(f_expr):
                print(f"(topo_order, op_type) : {f._topological_order}, {f_expr.op}")
            else:
                print(f"(topo_order, op_type) : {f._topological_order}, {type(f_expr)}, Non-call node")

            n_match_frontier = 0
            for pat in self._backendop_lib.get_all_patterns():
                # print("Checking... ", pat)

                # ordered_pattern_matcher consider the order of arguments when matching
                # in contrast to basic Relay pattern matching.
                # If we don't use this, we need to modify extract_subgraph (for op measurement)
                if self._ordered_pattern_matcher.match(f_expr, pat.get_relay_pattern()):
                # if pat.get_relay_pattern().match(f_expr):
                    assert pat.get_depth() >= 1 # 0 depth doesn't make sense
                    print("The following pattern is matched:", pat.get_relay_pattern())

                    # Get best backend op and its cost for matched nodes
                    best_backend_op, min_cost = get_optimal_backendop(self._backendop_lib, f_expr,
                                                                      pat, self._target_backend, hw_name)

                    # Skip update if there is no backend op available for matched pattern
                    if best_backend_op is None:
                        continue

                    # Extract match information; refer to detailed explanation in the MatchInfoExtractor
                    best_backend_op_name = repr(best_backend_op)
                    matched_nodes, match_dic, new_frontiers = extractor.extract(f_expr, pat.get_relay_pattern(), best_backend_op_name)

                    dp_table.update(matched_nodes, match_dic, best_backend_op_name, min_cost, new_frontiers)
                    # print(dp_table._dp_table)

                    # Add new frontiers to the queue
                    prev_qsize = frontier_queue._frontiers.qsize()
                    frontier_queue.put(new_frontiers)
                    n_match_frontier += frontier_queue._frontiers.qsize() - prev_qsize

            # if n_match_frontier == 0:
            #     printe("[Debug!] This frontier didn't match!!")
            # else:
            #     printe(f"n_match_froniter : {n_match_frontier}")

        # Assign backend operator annotation (group_id + backend_op_name) to Relay expr (backend attribute)
        dp_table.assign_backend_op_to_expr()


class AssignBackendExprVisitor:
    def __init__(self):
        self._memo = {}

    def assign(self, expr, annotation):
        self._memo = {}
        self._annotation = annotation
        self.visit_expr(expr)

    # Visit Relay expressions in post-order
    def visit_expr(self, expr):

        if hash(expr) in self._memo:
           return
        else:
            # memorize this visit to prevent it from visiting twice
            self._memo[hash(expr)] = True
            relay.analysis.update_backend(expr, self._annotation)

        # We assume that child class at least have methods for these
        if is_constant_node(expr):
            self.visit_expr_const(expr)
        elif is_var_node(expr):
            self.visit_expr_var(expr)
        elif is_tuplegetitem_node(expr):
            self.visit_expr_tuplegetitem(expr)
        elif is_call_node(expr):
            self.visit_expr_call(expr)
        elif is_function_node(expr):
            self.visit_expr_func(expr)
        elif is_tuple_node(expr):
            self.visit_expr_tuple(expr)
        else:
            raise Exception(f"Unexpected expression type, {type(expr)}")

    def visit_expr_const(self, expr):
        pass

    def visit_expr_var(self, expr):
        pass

    def visit_expr_tuple(self, expr):
        for arg in expr.fields:
            self.visit_expr(arg)

    def visit_expr_tuplegetitem(self, expr):
        self.visit_expr(expr.tuple_value)

    def visit_expr_call(self, expr):
        op, args, attrs, type_args, span = expr.op, expr.args, expr.attrs, expr.type_args, expr.span

        for arg in args:
            self.visit_expr(arg)

    def visit_expr_func(self, expr):
        params, body, ret_type, type_params = expr.params, expr.body, expr.ret_type, expr.type_params
        self.visit_expr(body)

# """
# FrontierGraph and FrontierNode is to allow effective search over matched backend op assignments.
# We use DFS to explore all possible combinations of backend ops.
# Given the maximum width of graph, DFS is more memory-efficient than BFS.
# """
# class FrontierNode:
#     def __init__(self, expr, backend_ops):
#         self.children = []
#         # Relay_expr is before matched backend_ops
#         self.relay_expr = expr
#         self.backend_ops = backend_ops
#
#     def add_child(self, child):
#         self.children.append(child)
#
#     def get_n_children(self):
#         return len(self.children)
#
# class FrontierGraph:
#     def __init__(self, root):
#         self.root = root

"""
We have two separate passes:
1) First pass is for generating backend op assignment tree by matching every possible backend op over graph.
We call this tree as FrontierGraph and it allows exhaustive search over all possible backend op assignments.
2) Second pass is for evaluating all possible backend op assignments for a graph. Note that we only compile an
entire graph rather than compiling multiple subgraphs to evaluate entire graph in DP.
"""
# class ExhaustiveSearcher:
#     def __init__(self, backendop_lib, target_backend=None):
#         self._backendop_lib = backendop_lib
#         self._target_backend = target_backend
#
#         # Attribute key to pass to N-to-1 lowering pass
#         self._bop_attr_key = "backend-op"
#
#         # For printing matched backend ops in ResNet graph
#         patterns = self._backendop_lib.get_all_patterns()
#         self._pattern_to_name = {}
#         for pat in patterns:
#             backend_ops = self._backendop_lib.pattern_to_backendops[pat]
#
#             assert len(backend_ops) > 0
#             name = backend_ops[0]._op_type.name()
#             self._pattern_to_name[pat] = name
#
#         self.memo_map = {}
#
#         # Key: Expr / Value: backend op id + name
#         self._backend_op_id = 0
#         self.frontier_graph = None
#
#     def debug_print(self, f, f_expr):
#         print("=" * 45)
#         if is_call_node(f_expr):
#             print(f"(topo_order, op_type) : {f._topological_order}, {f_expr.op}")
#         else:
#             print(f"(topo_order, op_type) : {f._topological_order}, {type(f_expr)}, Non-call node")
#
#     def create_backend_op_annotation(self, backend_op):
#         return f"{self._backend_op_id}-{backend_op}"
#
#     def get_backend_ops(self, pattern):
#         backend_ops = self._backendop_lib.get_backendops(pattern)
#         backend_op_annotations = []
#         for op in backend_ops:
#             if op.get_target() in self._target_backend:
#                 backend_op_annotations.append(self.create_backend_op_annotation(op))
#
#         return backend_op_annotations
#
#     def optimize(self, comp_graph):
#         # HACKY: Reset matched_expr
#         comp_graph.reset()
#
#         # Note that we have Node object (not Relay Expr) in the frontiers
#         frontiers = Q.PriorityQueue()
#         frontiers.put(comp_graph.get_root())
#         self.frontier_graph = FrontierGraph(FrontierNode(comp_graph.get_root().get_relay_expr(), None))
#         pair2match = {}
#
#         while not frontiers.empty():
#             # Facilitate the debugging process
#             self._backendop_lib.save_to_log(hw_name)
#
#             f = frontiers.get()
#             f_expr = f.get_relay_expr()
#             f_node = FrontierNode(comp_graph.get_root().get_relay_expr(), None)
#
#             # Debug printing
#             self.debug_print(f, f_expr)
#
#             # print(self._backendop_lib.get_all_patterns())
#             for pat in self._backendop_lib.get_all_patterns():
#                 # print(pat)
#                 if pat.get_relay_pattern().match(f_expr):
#                     # Check if there is an existing frontier with the same goal idx
#                     # Conv(Data, Weight)
#                     # get_next_expr_after_match -> [Data, Weight]
#                     # prev_expr_after_match = Conv()
#                     assert pat.get_deptn() >= 1
#
#                     tuple_after_matches = get_next_expr_after_match(f_expr, None, pat.get_relay_pattern())
#                     print("The following pattern is matched:", pat.get_relay_pattern())
#
#                     # Consdier only valid nodes
#                     tuple_after_matches = [tup for tup in tuple_after_matches if hash(tup[0]) in comp_graph.expr2node]
#                     for t_idx, (expr_after_match, prev_expr_after_match) in enumerate(tuple_after_matches):
#                         # Get new frontier, matched backend ops, and their costs
#                         new_loc = comp_graph.expr2node[hash(expr_after_match)]
#                         backend_ops = self.get_backend_ops(pat)
#
#                         # Skip update if there is no backend op available for matched pattern
#                         if len(backend_ops) == 0:
#                             continue
#
#                         # Flush matchings from second branch if there are more than one branches
#                         if t_idx == 0:
#                             pass
#                             # new_string = self.loc2match[hash(f)]['string'] + "-" + self._pattern_to_name[pat]
#                             # print(f"Assign matched op : {pat_op}")
#                         else:
#                             new_match, new_cost, new_string = [], 0, "+"
#
#                         # Maintain pair2match for keeping track of match results for each branch
#                         new_loc.matched_expr[hash(prev_expr_after_match)] = 1
#                         # new_loc is node after match
#                         out_key, in_key = hash(new_loc), hash(prev_expr_after_match)
#
#                         if out_key not in pair2match:
#                             pair2match[out_key] = {}
#
#                         if in_key not in pair2match[out_key] or pair2match[out_key][in_key]["cost"] > new_cost:
#                             pair2match[out_key][in_key] = {"match": new_match, "cost": new_cost, "string": new_string}
#
#                         # Update loc2match for final outcome
#                         if new_loc.get_n_parents() == new_loc.get_n_matched():
#                             if hash(new_loc) not in self.loc2match:
#                                 frontiers.put(new_loc)
#
#                             new_match, new_cost, new_string = [], 0, ""
#                             for _, match_dic in pair2match[out_key].items():
#                                 new_match += match_dic["match"]
#                                 new_cost += match_dic["cost"]
#                                 new_string += match_dic["string"]
#
#                             if hash(new_loc) not in self.loc2match or self.loc2match[hash(new_loc)]["cost"] > new_cost:
#                                 self.loc2match[hash(new_loc)] = {"match": new_match, "cost": new_cost,
#                                                                  "string": new_string}
#
#     def get_optimized_match(self, comp_graph):
#         assert self.loc2match is not None
#
#         # Get final match (expr, backend_op_name)
#         result_idx = -1
#         final_match = {}
#         fused_group_id = 0
#         # print(self.loc2match)
#         # print([hash(node) for node in comp_graph._nodes])
#         # print(comp_graph._nodes[-1].get_relay_expr())
#         for (pat_op, pat_cost, hash_expr) in self.loc2match[hash(comp_graph._nodes[result_idx])]["match"]:
#             final_match[hash_expr] = (fused_group_id, pat_op)
#             fused_group_id += 1
#
#         optimized_match, post_order_match_result = ExprMatcher(final_match).match(comp_graph.get_relay_expr())
#
#         return optimized_match, post_order_match_result
#
