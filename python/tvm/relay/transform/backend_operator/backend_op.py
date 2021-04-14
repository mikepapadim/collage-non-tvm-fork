import sys

from enum import Enum, auto
from dataclasses import dataclass
from collections import defaultdict
from functools import lru_cache

import numpy as np

import tvm
from tvm.relay.dataflow_pattern import *
from tvm import relay
import tvm.relay.testing as testing
from pathlib import Path

from .pattern import Pattern
from .utils import get_diamond
from .utils import is_call_node, is_tuplegetitem_node, is_var_node, no_constraints_func, is_constant_node
from .op_config import Config, MeasuredConfigs
from .target import Target, get_target_cost_func
from .op_type import OpType, optype_to_pattern, relayop_to_varnames

# It gives the path of backend_op.py no matter where you import this file
# cur_dir_path = Path(__file__).parent.absolute()
# RES_LOG = f"{cur_dir_path}/logs/runtime_results.log"

# redirect stdout to this log so it is not intertwined with by TVM backend log output
# sys.stdout = open(RES_LOG, 'w')

# TODO: here we are delegating TVM to choose the appropriate backend operator given the target and an expr,
# maybe we want more fine-grained control

class BackendOp(object):
  def __init__(self, name, target, op_type, max_depth, measured_configs_lib, constraint_func):
    self._name = name
    self._target = target
    self._op_type = op_type
    self._max_depth = max_depth
    self._pattern = optype_to_pattern[op_type]
    self._measured_configs = measured_configs_lib
    self._constraint_func = constraint_func


  def __repr__(self):
    return self._name

  def get_pattern(self):
    return self._pattern

  def get_target(self):
    return self._target

  # max depth is the depth of the longest branch
  # (chain pattern has a single branch, diamond pattern has 2 branchese)
  def get_max_depth(self):
    return self._max_depth

  def get_cost(self, expr):

    # configuration: backend operator name, operator type (which encode pattern), data shape, node attributes

    config = Config(self._name, self._op_type.name(), expr)
    # print(config)

    # if constraints are not satisfied, return infinite cost
    if not self._constraint_func(config):
      return float('inf')

    cost_info = self._measured_configs.get_cost(config)
    if cost_info != None:
      # pass
      print("REUSED RESULT!!!!")
    else:
        # print("!!!Warning!!! Random cost!")
        # cost_info = (-1, -1)
      print("NOT REUSED!!!!")
    
      cost_func = get_target_cost_func(self._target)
      cost_info = cost_func(self._name, expr, self._target)
      self._measured_configs.save_cost(config, cost_info)
    
    # We use mean as a cost instead of sampling for now
    mean_cost, std_cost = cost_info
    return mean_cost

# extract the subgraph of the expr that matches the pattern (only the top layers of the recursive relay expr).
# Since there might be multiple branches, we traverse each branch by "max_depth" steps, and rewrite the child nodes
# of the last node to free variables. However, when there are multiple branches, only the rewrite at the end of the
# longest branch will be useful
# Additionally, we use a cache to memoize calculated results
@lru_cache(None)
def extract_subgraph(expr, max_depth):
  # use this to make sure the nodes (exprs) are consistent between different branches
  old_expr_to_new = dict()

  def get_expr(expr):
    if expr in old_expr_to_new:
      return old_expr_to_new[expr]
    return expr

  def helper(expr, depth):
    expr = get_expr(expr)

    if is_call_node(expr):
      # note that only call node has "op" attribute corresponding to a single backend operator
      op, args, attrs, type_args, span = expr.op, expr.args, expr.attrs, expr.type_args, expr.span

      new_args = []
      # at depth 1, turn call expr arguments into free variables with the same attributes and data shapes!
      if depth == 1:
        var_names = relayop_to_varnames[op.name]
        
        # # of arguments should match # of type arguments
        # Fix: This happens in BERT. We need to deal with it
        # It means that type inference hasn't been executed (type_args are not filled)
        # because inputs are variables, not relay op expr
        if len(expr.args) != len(expr.type_args):
          raise Exception("The type inference pass hasn't been executed.")
        else:
          for i in range(len(expr.args)):
            type_arg = expr.type_args[i]
            var_name = var_names[i]

            # Bias should be constant
            if var_name == 'bias':
              input_data = expr.args[i].data
              new_args.append(relay.Constant(input_data))
            else:
              new_args.append(relay.var(var_name, type_arg))
      else:
        for child in expr.args:
          new_args.append(helper(child, depth - 1))

      args = new_args

      new_expr = tvm.relay.expr.Call(op, args, attrs, type_args, span)
      old_expr_to_new[expr] = new_expr
      return new_expr

    elif is_tuplegetitem_node(expr):
      tuple_value = helper(expr.tuple_value, depth)
      new_expr = tvm.relay.expr.TupleGetItem(tuple_value, expr.index)
      old_expr_to_new[expr] = new_expr
      return new_expr

    elif is_var_node(expr):
      return expr

    elif is_constant_node(expr):
      return expr

    else:
      raise Exception("Expr type not implemented")

  return helper(expr, max_depth)

# given a pattern and a relay expr matching that pattern, return the cheapest backend operator
# satisfying the constraints and its cost. Return None if no backend operators satisfy constraints.
def get_optimal_backendop(b_op_lib, expr, pattern, target = None):
  assert type(target) == list
  
  backendops = b_op_lib.get_backendops(pattern)

  cheapest_op, min_cost = None, float('inf')
  for op in backendops:
    # if target is not None, only consider backend operators for that target
    if target != None and op.get_target() not in target:
      continue

    max_depth = op.get_max_depth()
    subgraph = extract_subgraph(expr, max_depth)
    # print("Subgraph: ", subgraph)
    cost = op.get_cost(subgraph)

    assert cost != None
    if cost < min_cost:
      min_cost = cost
      cheapest_op = op

  if min_cost == float('inf'):
    raise Exception("No corresponding backend operators")
    return None
  return cheapest_op, min_cost




