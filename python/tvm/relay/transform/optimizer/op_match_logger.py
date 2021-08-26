from tvm.relay.expr_functor import ExprVisitor
from ..backend_operator.target import USER_DEFINED_MATCH_LOG
from ..backend_operator.utils import *
import pandas as pd
from tvm.ir import Op
from ..utility.debug_helper import printe

DUMMY_VAL = 0
COL_NAME = 'annotation'

"""
** Use this when you need match information in different build call;
Within the same build call, pass annotation through expr.backend instead of logging

Opretaor match logger
- dump and optimized_match (Key: expression / Value: annotation (group_id + op_name))
- Log file is located in the f"transform/logs/{USER_DEFINED_MATCH_LOG}"
- In the log, each line represents one match betweeen backend operator and node
    e.g., 0,1-tensorrt_conv2d+relu
    > 0 means post_dfs_order
    > 1 - group id
    > tensorrt_conv2d+relu - backend operator name
    > Note that 1-tensorrt_conv2d+relu is the annotation of match.

"""
class OpMatchLogger(ExprVisitor):
    def __init__(self):
        super().__init__()

    # Dump optimized_match into a log file in csv format using dataframe
    def save(self, expr, optimized_match, log_path = USER_DEFINED_MATCH_LOG):
        assert not is_function_node(expr)

        self.memo_map = {}
        self._optimized_match = optimized_match

        # For logging
        self._memo_map_for_log = {}
        self._log_dic = {}
        self._post_dfs_order = 0
        self.visit(expr)

        # Create dataframe and dump into csv
        df = pd.DataFrame.from_dict(self._log_dic, orient="index")

        # For better printing
        df.columns = [COL_NAME]
        df.index.name = "post_dfs_order"
        df.to_csv(log_path)
        printe(f"Log is saved to {log_path}")
        # printe(df)

    # Visit Relay expressions in post-order
    def visit(self, expr):
        super().visit(expr)

        if expr in self._memo_map_for_log:
            return
        else:
            # Op should be skipped
            if not isinstance(expr, Op):
                assert expr in self._optimized_match

                anno = self._optimized_match[expr]
                self._log_dic[self._post_dfs_order] = anno
                relay.analysis.update_backend(expr, anno)

                # print(self._post_dfs_order, anno)
                self._post_dfs_order += 1

        self._memo_map_for_log[expr] = DUMMY_VAL

"""
** Use this when you need match information in different build call;
Within the same build call, pass annotation through expr.backend instead of logging

Opretaor match logger
- dump and optimized_match (Key: expression / Value: annotation (group_id + op_name))
- Log file is located in the f"transform/logs/{USER_DEFINED_MATCH_LOG}"
- In the log, each line represents one match betweeen backend operator and node
    e.g., 0,1-tensorrt_conv2d+relu
    > 0 means post_dfs_order
    > 1 - group id
    > tensorrt_conv2d+relu - backend operator name
    > Note that 1-tensorrt_conv2d+relu is the annotation of match.

"""


class OpMatchReader(ExprVisitor):
    def __init__(self):
        super().__init__()

    # read csv file and translate it into opt_match dictionary
    # (Key: expression / Value: annotation (group_id + op_name))
    def read(self, expr, log_path = USER_DEFINED_MATCH_LOG):
        assert not is_function_node(expr)
        df = pd.read_csv(log_path, index_col=0)
        # print(list(df.index))
        # print(list(df['annotation']))

        self._post_dfs_order_to_anno = df.to_dict()[COL_NAME]

        # Translate dataframe into opt_match dictioanry
        self._memo_map_for_log = {}
        self.opt_match_from_log = {}
        self._post_dfs_order = 0
        self.visit(expr)

        return self.opt_match_from_log

    # Visit Relay expressions in post-order
    def visit(self, expr):
        super().visit(expr)

        if expr in self._memo_map_for_log:
            return
        else:
            # Op should be skipped
            if not isinstance(expr, Op):
                assert self._post_dfs_order in self._post_dfs_order_to_anno

                anno = self._post_dfs_order_to_anno[self._post_dfs_order]
                self.opt_match_from_log[expr] = anno
                relay.analysis.update_backend(expr, anno)
                self._post_dfs_order += 1

        self._memo_map_for_log[expr] = DUMMY_VAL