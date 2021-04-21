import tvm
from tvm import relay
from tvm.contrib import graph_runtime as runtime
import numpy as np

from .optimizer_utils import get_pattern_len, get_next_expr_after_match
from ..backend_operator.utils import is_call_node, is_tuplegetitem_node, is_var_node, is_constant_node, is_function_node
from ..backend_operator.backend_op import get_optimal_backendop

from ..backend_operator.utils import get_data_shape

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
            self.visit_expr_tuple(expr, annotation)
            node_type = "TupleGetItem"
        elif is_call_node(expr):
            self.visit_expr_call(expr, annotation)
            node_type = expr.op
        elif is_function_node(expr):
            self.visit_expr_func(expr, annotation)
            node_type = "Function"
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
            self._optimized_match[expr] = f"{annotation[0]}-{annotation[1]}"
            self._topo_order_to_op.append((node_type, self._optimized_match[expr]))
        else:
            raise Exception("Expression should not be visited more than once")

        # print("After each expression", self._optimized_match)

    def visit_expr_const(self, expr, annotation):
        pass
        
    def visit_expr_var(self, expr, annotation):
        pass
    
    def visit_expr_tuple(self, expr, annotation):
        self.visit_expr(expr.tuple_value, annotation)
    
    def visit_expr_call(self, expr, annotation):
        op, args, attrs, type_args, span = expr.op, expr.args, expr.attrs, expr.type_args, expr.span
        
        for arg in args:
            self.visit_expr(arg, annotation)

    def visit_expr_func(self, expr, annotation):
        params, body, ret_type, type_params = expr.params, expr.body, expr.ret_type, expr.type_params
        self.visit_expr(body, annotation)

# class ExprVisitor:
#     def __init__(self):
#         self._memo = {}
#
#     def visit(self, expr):
#         return self.visit_expr(expr)
#
#     # Visit Relay expressions in post-order
#     def visit_expr(self, expr):
#         # We assume that child class at least have methods for these
#         print(repr(expr))
#
#         if is_constant_node(expr):
#             self.visit_expr_const(expr)
#         elif is_var_node(expr):
#             self.visit_expr_var(expr)
#         elif is_tuplegetitem_node(expr):
#             self.visit_expr_tuple(expr)
#         elif is_call_node(expr):
#             self.visit_expr_call(expr)
#         elif is_function_node(expr):
#             self.visit_expr_func(expr)
#         else:
#             raise Exception(f"Unexpected expression type, {type(expr)}")
#
#     def visit_expr_const(self, expr):
#         pass
#
#     def visit_expr_var(self, expr):
#         pass
#
#     def visit_expr_tuple(self, expr):
#         self.visit_expr(expr.tuple_value)
#
#     def visit_expr_call(self, expr):
#         for arg in expr.args:
#             if hash(arg) not in self._memo:
#                 # memorize this visit to prevent it from visiting twice
#                 new_arg = self.visit_expr(arg)
#                 self._memo[hash(arg)] = True
#
#     def visit_expr_func(self, expr):
#         self.visit_expr(expr.body)

    
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
        
        # For Function inputs renaming (recreation)
        self._local_memo = None
        self._func_var_id = -1
        self._has_call = 0
        
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
            if is_call_node(f_expr):
                print(f"(topo_order, op_type) : {f._topological_order}, {f_expr.op}")
            else:
                print(f"(topo_order, op_type) : {f._topological_order}, {f_expr}, Non-call node")
            
            # print(self._backendop_lib.get_all_patterns())
            for pat in self._backendop_lib.get_all_patterns():
                # print(pat)
                if pat.get_pattern().match(f_expr):
                    # Check if there is an existing frontier with the same goal idx
                    # Conv(Data, Weight)
                    # get_next_expr_after_match -> [Data, Weight]
                    # next_expr_after_match = Conv()
                    assert get_pattern_len(pat.get_pattern()) >= 1
                    tuple_after_matches = get_next_expr_after_match(f_expr, None, get_pattern_len(pat.get_pattern()))
                    print("PATTERN MATCHED", pat.get_pattern())
                    # Consdier only valid nodes
                    tuple_after_matches = [tup for tup in tuple_after_matches if hash(tup[0]) in comp_graph.expr2node]
                    for t_idx, (expr_after_match, prev_expr_after_match) in enumerate(tuple_after_matches):
                        # Get new frontier, matched backend ops, and their costs
                        new_loc = comp_graph.expr2node[hash(expr_after_match)]
                        pat_op, pat_cost = get_optimal_backendop(self._backendop_lib, f_expr, pat, self._target_backend)

                        # new_match = self.loc2match[hash(f)]["match"] + [(pat_op, pat_cost, hash(f_expr))]
                        # new_cost = self.loc2match[hash(f)]["cost"] + pat_cost
                        # new_string = self.loc2match[hash(f)]['string'] + "-" + self._pattern_to_name[pat]

                        # Flush matchings from second branch if there are more than one branches
                        if t_idx == 0:
                            new_match = self.loc2match[hash(f)]["match"] + [(pat_op, pat_cost, hash(f_expr))]
                            new_cost = self.loc2match[hash(f)]["cost"] + pat_cost
                            new_string = self.loc2match[hash(f)]['string'] + "-" + self._pattern_to_name[pat]
                            print(f"Assign matched op : {pat_op}")
                        else:
                            new_match, new_cost, new_string = [], 0, "+"

                        # Maintain pair2match for keeping track of match results for each branch
                        new_loc.matched_expr[hash(prev_expr_after_match)] = 1
                        out_key = hash(new_loc) # new_loc is node
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

                            if hash(new_loc) not in self.loc2match or self.loc2match[hash(new_loc)]["cost"] > new_cost:
                                self.loc2match[hash(new_loc)] = {"match": new_match, "cost":new_cost, "string":new_string}
    
#     def _change_func_var(self, expr):
        
#         var_arr = []
        
#         if type(expr) == tvm.relay.Constant:
#             self._func_var_id += 1
#             new_expr = tvm.relay.var(f"p{self._func_var_id}", tvm.relay.TensorType(expr.data.shape, 'float32'))
#             var_arr.append(new_expr)
#         elif is_var_node(expr):
#             self._func_var_id += 1
#             new_expr = tvm.relay.var(f"p{self._func_var_id}", expr.type_annotation)
#             var_arr.append(new_expr)
#         elif type(expr) == tvm.relay.Function:
# #             new_expr = expr
# #             var_arr += new_expr.params
#             raise Exception("Relay Function shouldn't be visted.")
#         elif is_tuplegetitem_node(expr):
#             new_expr = self._change_func_var(expr.tuple_value)
#             new_expr = tvm.relay.expr.TupleGetItem(matched_expr, expr.index)
#         elif is_call_node(expr):
#             self._has_call = 1
#             op, args, attrs, type_args, span = expr.op, expr.args, expr.attrs, expr.type_args, expr.span
            
#             new_args = []
#             for arg in args:
#                 if hash(arg) not in self._local_memo:
#                     # memorize this visit to prevent it from visiting twice
#                     new_arg, arg_var_arr = self._change_func_var(arg)
#                     self._memo[hash(arg)] = new_arg, arg_var_arr
#                 else:
#                     new_arg, arg_var_arr  = self._memo[hash(arg)]
                
#                 var_arr += arg_var_arr
#                 new_args.append(new_arg)
            
#             new_expr = tvm.relay.expr.Call(op, new_args, attrs, type_args, span)
#         else:
#             raise Exception("Expr type not implemented")

#         return new_expr, var_arr
    
    # We can replace var_arr with relay.analysis.free_vars(expr)
#     def _construct_matched_expr(self, expr, match):
#         matched_expr = None
#         type_args = None
#         var_arr = []
        
#         if is_var_node(expr):
#             matched_expr = expr
#             var_arr.append(expr)
#         elif type(expr) == tvm.relay.Constant:
#             matched_expr = expr
#             var_arr.append(expr)
#         elif is_tuplegetitem_node(expr):
#             matched_expr, var_arr = self._construct_matched_expr(expr.tuple_value, match)
#             matched_expr = tvm.relay.expr.TupleGetItem(matched_expr, expr.index)
#         elif is_call_node(expr):
#             op, args, attrs, type_args, span = expr.op, expr.args, expr.attrs, expr.type_args, expr.span
            
#             matched_args = []
#             new_arg = None
#             for arg in args:
#                 if hash(arg) not in self._memo:
#                     # memorize this visit to prevent it from visiting twice
#                     new_arg, arg_var_arr = self._construct_matched_expr(arg, match)
#                     self._memo[hash(arg)] = new_arg, arg_var_arr
#                 else:
#                     new_arg, arg_var_arr = self._memo[hash(arg)]
                
#                 var_arr += arg_var_arr
#                 matched_args.append(new_arg)
            
#             matched_expr = tvm.relay.expr.Call(op, matched_args, attrs, type_args, span)
#         else:
#             raise Exception("Expr type not implemented")
    
#         # Annotate backend operator by converting expr to function
#         if hash(expr) in match:
#             backend_op_name = repr(match[hash(expr)])            
            
#             self._local_memo = {}
#             self._has_call = 0
            
#             matched_expr, func_var_arg = self._change_func_var(matched_expr)
# #             func_var_arg = relay.analysis.free_vars(matched_expr)
#             matched_expr = tvm.relay.Function(func_var_arg, matched_expr).with_attr(self._bop_attr_key, backend_op_name)
#             matched_expr = matched_expr.with_attr("Primitive", self._has_call)
#             matched_expr = tvm.relay.expr.Call(matched_expr, var_arr)
        
#         return matched_expr, var_arr
           
    def get_optimized_match(self, comp_graph):
        assert self.loc2match is not None
        
        # Get final match (expr, backend_op_name)
        result_idx = -1
        final_match = {}
        fused_group_id = 0
        for (pat_op, pat_cost, hash_expr) in self.loc2match[hash(comp_graph._nodes[result_idx])]["match"]:
            final_match[hash_expr] = (fused_group_id, pat_op)
            fused_group_id += 1

        optimized_match, post_order_match_result = ExprMatcher(final_match).match(comp_graph.get_relay_expr())

        return optimized_match, post_order_match_result
        
        
        
        

        
