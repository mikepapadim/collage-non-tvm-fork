from tvm.relay.expr_functor import ExprVisitor
from collections import defaultdict
from ..backend_operator.utils import *
from tvm.ir import Op

"""
This class translates optimized_match (Key: expression / Value: annotation (group_id + op_name))
into state_id_to_exprs_anno (Key: state id / Value: a list of pairs of expression and its annotation)

state_id_to_exprs_anno will be used for building state (search space) for evolutionary search 
"""

class MatchToOpStateTranslator(ExprVisitor):
    def __init__(self):
        super().__init__()

    def translate(self, expr, optimized_match):
        self.memo_map = {}
        self._optimized_match = optimized_match
        self.state_id_to_exprs_anno = defaultdict(list)
        self.visit(expr)

        return self.state_id_to_exprs_anno

    # Visit Relay expressions in post-order
    def visit(self, expr):
        if expr in self.memo_map:
            return
        else:
            # Op should be skipped
            if not isinstance(expr, Op):
                assert expr in self._optimized_match

                anno = self._optimized_match[expr]
                # Note that group id is passed as a string
                group_id = int(get_group_id_from_backend_op_annotation(anno))
                backend_op_name = get_backendop_name_from_backend_op_annotation(anno)

                # Group id is same as state id
                self.state_id_to_exprs_anno[group_id].append((hash(expr), backend_op_name))

        super().visit(expr)
#
# class OpState:
#     def __init__(self, state_id, exprs, annotation):
#         self._state_id =  state_id