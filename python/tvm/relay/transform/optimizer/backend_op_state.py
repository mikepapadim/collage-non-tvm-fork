from tvm.relay.expr_functor import ExprVisitor
from collections import defaultdict
from ..backend_operator.utils import *
from tvm.ir import Op
import copy
from enum import IntEnum

EXT_COMPILERS = ["tensorrt"]

class BackendStateType(IntEnum):
    # This is for measurement
    FIRST_LEVEL_BEST_OP = 0
    TENSORRT_OP = 1

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

"""
This class translates optimized_match (Key: expression / Value: annotation (group_id + op_name))
into group_id_to_exprs_anno (Key: state id / Value: a list of pairs of expression and its annotation)

group_id_to_exprs_anno will be used for building state (search space) for evolutionary search 
"""

class MatchToOpGroupTranslator(ExprVisitor):
    def __init__(self):
        super().__init__()

    def translate(self, expr, optimized_match):
        assert not is_function_node(expr)

        self.memo_map = {}
        self._optimized_match = optimized_match
        self.group_id_to_exprs_anno = defaultdict(list)
        self.visit(expr)

        return self.group_id_to_exprs_anno

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
                # backend_op_name = get_backendop_name_from_backend_op_annotation(anno)

                # Group id is same as state id
                self.group_id_to_exprs_anno[group_id].append((expr, anno))

        super().visit(expr)


"""
This back-translates state_id_to_exprs_anno into optimized_match for measurement; 
It translates state_id_to_exprs_anno (Key: state id / Value: a list of pairs of expression and its annotation)
into optimized_match (Key: expression / Value: annotation (group_id + op_name)) 
"""

class OpStateToMatchTranslator():
    def __init__(self, optimized_match, group_id_to_exprs_anno):
        self.optimized_match = optimized_match
        self.group_id_to_exprs_anno = group_id_to_exprs_anno
        self.state_id_to_group_id = self.get_valid_op_state_by_filtering()

    def is_valid_op_state(self, expr_anno_pairs):
        assert len(expr_anno_pairs) > 0

        # To check if ops in each group have same backend
        expr, anno = expr_anno_pairs[0]
        first_backend = get_backend_from_backend_op_annotation(anno)
        is_valid_op_state = True

        for expr, anno in expr_anno_pairs:
            # To check if ops in each group have same backend
            backend_name = get_backend_from_backend_op_annotation(anno)
            assert backend_name == first_backend

            # If one of ops is tuple or tuple_get_item, then prevent it from being ext compiler ops
            if is_tuple_node(expr) or is_tuplegetitem_node(expr):
                is_valid_op_state = False
                break
            # print(type(expr), anno)

        # If this is TensorRT op chosen from the first op optimizing pass,
        # Then we don't need to consider it on the second subgraph optimizing pass.
        if first_backend in EXT_COMPILERS:
            is_valid_op_state = False

        # print(is_valid_op_state)
        # print("-"*30)
        return is_valid_op_state


    # Valid op states include ops that are not assigned to TensorRT
    # and that are not Tuple or TupleGetItem (These shouldn't be TensorRT ops)
    def get_valid_op_state_by_filtering(self):
        state_id_to_group_id = {}
        state_id = 0
        group_ids = self.group_id_to_exprs_anno.keys()
        for group_id in group_ids:
            expr_anno_pairs = self.group_id_to_exprs_anno[group_id]
            if self.is_valid_op_state(expr_anno_pairs):
                state_id_to_group_id[state_id] = group_id
                state_id += 1
        # print(state_id_to_group_id)
        # sys.exit(0)
        return state_id_to_group_id

    def gen_trt_annotation(self, anno):
        group_id = int(get_group_id_from_backend_op_annotation(anno))
        op_name = get_op_name_from_backend_op_annotation(anno)
        backend_name = 'tensorrt'

        return f"{group_id}-{backend_name}_{op_name}"

    # It only changes op to TensorRT op now
    def update_opt_match(self, state_id, new_opt_match):
        # print(new_opt_match)
        gid = self.state_id_to_group_id[state_id]
        for expr, anno in self.group_id_to_exprs_anno[gid]:
            assert expr in new_opt_match
            new_opt_match[expr] = self.gen_trt_annotation(anno)
            # print(f"anno / new anno: {anno} / {new_opt_match[expr]}")

    def translate(self, op_state):
        # Warning(@Soo): if you do deepcopy, it will create new instances copied from elements of list or dict
        new_opt_match = self.optimized_match.copy()

        # Translate individual into match
        # gid represents op group id (or op id)
        # backend_type represents 0 or 1
        # 0 - backend op selected from first level
        # 1 - tensorrt op
        for state_id, backend_type in enumerate(op_state):
            # print(backend_type)
            if backend_type == BackendStateType.TENSORRT_OP:
                # print("tensorrt")
                self.update_opt_match(state_id, new_opt_match)

        # sys.exit(0)

        return new_opt_match
#
# class OpState:
#     def __init__(self, state_id, exprs, annotation):
#         self._state_id =  state_id