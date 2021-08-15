from ..backend_operator.utils import *

class OrderedPatternMatcher:
    def __init__(self):
        pass

    def match(self, expr, pattern):
        # self.matched_exprs = []
        is_matched = self.visit_expr(expr, pattern)
        return is_matched #, self.matched_exprs

    # Visit Relay expressions in post-order
    def visit_expr(self, expr, pattern):
        is_matched = True

        if not pattern.match(expr):
            return False

        if isinstance(pattern, WildcardPattern):
            return True

        # If the code reaches here, it meas that expr matches and the pattern is not Wildcard.
        # Thus, we should add this to matched_exprs if it's not constant or var node
        # Note that we only save operator nodes as matched exprs
        # if not is_constant_node(expr) and not is_var_node(expr):
        #     self.matched_exprs.append(expr)

        # We assume that child class at least have methods for these
        if is_constant_node(expr) or is_var_node(expr):
            pass
        elif is_tuplegetitem_node(expr):
            is_matched &= self.visit_expr_tuplegetitem(expr, pattern)
        elif is_call_node(expr):
            is_matched &= self.visit_expr_call(expr, pattern)
        elif is_tuple_node(expr):
            is_matched &= self.visit_expr_tuple(expr, pattern)
        else:
            raise Exception(f"Unexpected expression type, {type(expr)}")

        return is_matched

    def visit_expr_tuple(self, expr, pattern):
        is_matched = True
        for a_idx, arg in enumerate(expr.fields):
            is_matched &= self.visit_expr(arg, pattern.fields[a_idx])

        return is_matched

    def visit_expr_tuplegetitem(self, expr, pattern):
        is_matched = True
        is_matched &= self.visit_expr(expr.tuple_value, pattern.tuple_value)
        return is_matched

    def visit_expr_call(self, expr, pattern):
        is_matched = True
        op, args, attrs, type_args, span = expr.op, expr.args, expr.attrs, expr.type_args, expr.span

        for a_idx, arg in enumerate(args):
            is_matched &= self.visit_expr(arg, pattern.args[a_idx])

        return is_matched