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

from .pattern_language import Pattern
from .utils import get_diamond
from .utils import *
from .op_config import Config, MeasuredConfigs
from .cost_func import get_target_cost_func, Target
from .default_pattern import optype_to_pattern, relayop_to_varnames
from ..utility.debug_helper import printe
import logging

# It gives the path of backend_op.py no matter where you import this file
# cur_dir_path = Path(__file__).parent.absolute()
# RES_LOG = f"{cur_dir_path}/../logs/runtime_results.log"

# redirect stdout to this log so it is not intertwined with by TVM backend log output
# sys.stdout = open(RES_LOG, 'w')

# TODO: here we are delegating TVM to choose the appropriate backend operator given the target and an expr,
# maybe we want more fine-grained control

class BackendOp(object):
  def __init__(self, target, pattern, measured_configs_lib, constraint_func):
  #def __init__(self, name, target, op_name, depth, measured_configs_lib, constraint_func):
    self._target = target
    self._op_name = pattern.get_name()
    self._name = target.name() + "_" + self._op_name

    self._depth = pattern.get_depth()
    self._pattern = pattern #optype_to_pattern[pattern]
    self._measured_configs = measured_configs_lib
    self._constraint_func = constraint_func

  def __hash__(self):
      return hash((self._name, self._measured_configs, self._constraint_func))

  def __eq__(self, other):
      return self._name == other._name and self._measured_configs == other._measured_configs and self._constraint_func == other._constraint_func

  def __repr__(self):
    return self._name

  def get_pattern(self):
    return self._pattern

  def get_target(self):
    return self._target

  # max depth is the depth of the longest branch
  # (chain pattern has a single branch, diamond pattern has 2 branchese)
  def get_depth(self):
    return self._depth

  def get_cost(self, expr, hw_name):

    # configuration: backend operator name, operator type (which encode pattern), data shape, node attributes

    config = Config(self._name, self._op_name, expr)
    # print(config)

    # For Tuple, we do not need to measure it
    if is_tuple_node(expr) or is_tuplegetitem_node(expr):
      return 0#, 0

    # if constraints are not satisfied, return infinite cost
    if not self._constraint_func(config):
      return float('inf')#, 0

    cost_info = self._measured_configs.get_cost(config)

    # if self._target == Target.CUDNN and  self._op_name == "conv2d+biasadd+relu":
    #   cost_info = (0, 0)

    if cost_info != None:
      # pass
      logging.info("REUSED RESULT!!!!")
    else:
        # print("!!!Warning!!! Random cost!")
        # cost_info = (-1, -1)
      logging.info("NOT REUSED!!!!")

      cost_func = get_target_cost_func(self._target)
      cost_info = cost_func(self._name, expr, self._target, hw_name)
      self._measured_configs.save_cost(config, cost_info)

    # We use mean as a cost instead of sampling for now
    mean_cost, std_cost = cost_info
    return mean_cost

# extract the subgraph of the expr that matches the pattern (only the top layers of the recursive relay expr).
# Since there might be multiple branches, we traverse each branch by "depth" steps, and rewrite the child nodes
# of the last node to free variables. However, when there are multiple branches, only the rewrite at the end of the
# longest branch will be useful
# Additionally, we use a cache to memoize calculated results
#@lru_cache(None)
def extract_subgraph(expr, pattern):
  # use this to make sure the nodes (exprs) are consistent between different branches
  # Warning(@Soo): This is not necessary when we don't have diamond patterns
  # Still, this case happens in NasNet-A, e.g., addition of same avgpool2d results
  old_expr_to_new = dict()

  depth = pattern.get_depth()
  # print(f"depth: {depth}")

  relay_pattern = pattern.get_relay_pattern()
  def set_old_expr_to_new(expr, new_expr):
    old_expr_to_new[expr] = new_expr

  def get_expr(expr):
    if expr in old_expr_to_new:
      return old_expr_to_new[expr]
    return expr

  def helper(expr, depth, relay_pattern):
    assert relay_pattern.match(expr), f"(relay_pattern, expr) = ({relay_pattern}, {expr.op}, {expr.args[0].op})"
    # Warning(@Soo): To resolve NasNet-A corner case, e.g., addition of same avgpool2d results
    cur_checked_type = expr.checked_type
    expr = get_expr(expr)

    if isinstance(relay_pattern, WildcardPattern):
      # try:
      #   ret = relay.var("data", expr.checked_type)
      # except ValueError:
      #   # Warning(@Soo): Hacky exception handling for NasNet-A (same avgpool2d for two inputs of add)
      #   # The issue is that avgpool2d somehow is missing type information even after type inference      #
      #   printe("Checked type is not available for following expr")
      #   printe(repr(expr))

      # The above issue with avgpool2d is resolved!
      # The problem is because checked_type is updated when generating new expr.
      # We resolved it by saving checked_type from old expression
      ret = relay.var("data", cur_checked_type)
      return ret
    elif isinstance(relay_pattern, ConstantPattern):
      return expr
    elif is_call_node(expr):
      # note that only call node has "op" attribute corresponding to a single backend operator
      op, args, attrs, type_args, span = expr.op, expr.args, expr.attrs, expr.type_args, expr.span
      # if expr.op.name == 'reshape':
      #   print(f"New shape for reshape op : {expr.attrs.newshape}")
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
          # print(expr.op, var_names)
          for i in range(len(expr.args)):
            type_arg = expr.type_args[i]
            var_name = var_names[i]

            # Tuple should be treated separately
            if (type(type_arg) is tvm.ir.type.TupleType):
                input_data = expr.args[i]
                #print(type_arg.fields)
                new_args.append(relay.Tuple([relay.var(var_name, d) for i, d in enumerate(type_arg.fields)] ))
            # Note(@Soo): we get same perf no matter whether we use Var or Constant
            # elif var_name == 'weight':
            #   input_data = expr.args[i].data
            #   new_args.append(relay.Constant(input_data))
            # Bias should be constant
            elif var_name == 'bias':
              input_data = expr.args[i].data
              new_args.append(relay.Constant(input_data))
            else:
              new_args.append(relay.var(var_name, type_arg))
      else:
        for c_idx, child in enumerate(expr.args):
          pat_child = relay_pattern.args[c_idx]
          new_args.append(helper(child, depth - 1, pat_child))

      args = new_args

      new_expr = tvm.relay.expr.Call(op, args, attrs, type_args, span)
      set_old_expr_to_new(expr, new_expr)
      return new_expr

    elif is_tuple_node(expr):
      new_args = []
      for c_idx, child in enumerate(expr.fields):
        pat_child = relay_pattern.fields[c_idx]
        new_args.append(helper(child, depth - 1, pat_child))
      new_expr = relay.Tuple(new_args)
      set_old_expr_to_new(expr, new_expr)

      return new_expr

    elif is_tuplegetitem_node(expr):
      tuple_value = helper(expr.tuple_value, depth, relay_pattern.tuple_value)
      new_expr = tvm.relay.expr.TupleGetItem(tuple_value, expr.index)
      set_old_expr_to_new(expr, new_expr)
      return new_expr

    elif is_var_node(expr):
      return expr

    elif is_constant_node(expr):
      return expr

    else:
      raise Exception(f"Expr type not implemented {type(expr)}")

  return helper(expr, depth, relay_pattern)

# given a pattern and a relay expr matching that pattern, return the cheapest backend operator
# satisfying the constraints and its cost. Return None if no backend operators satisfy constraints.
def get_optimal_backendop(b_op_lib, expr, pattern, target = None, hw_name = "INVALID",
                          need_tvm_fallback_ops=False, fallback_backend_pats=None):
  assert type(target) == list

  backendops = b_op_lib.get_backendops(pattern)
  cheapest_op, min_cost = None, float('inf')

  for op in backendops:
    # if target is not None, only consider backend operators for that target
    if target != None and op.get_target() not in target:
      continue

    subgraph = extract_subgraph(expr, pattern)

    # if is_op("nn.conv2d")(wildcard(), wildcard()).match(expr):
    #   eprint(f"subgraph expr: {repr(subgraph)}")

    # Print the useful logs
    logging.info("-" * 45)
    # Warning(@Soo): there is a bug in printing repr of tuple in TVM.
    if is_tuple_node(subgraph):
      logging.info(f"Subgraph to measure (target: {str(op._target.name())}): {subgraph}")
    else:
      logging.info(f"Subgraph to measure (target: {str(op._target.name())}): {repr(subgraph)}")
    cost = op.get_cost(subgraph, hw_name)
    logging.info(f"Cost of subgraph : {cost:4f}")
    logging.info("-" * 45)

    assert cost != None
    if cost < min_cost:
      min_cost = cost
      cheapest_op = repr(op)

  # If no operator matched current patterns, fall back on TVM (no tuning) ops
  if cheapest_op is None:
    assert need_tvm_fallback_ops

    if pattern in fallback_backend_pats:
      # For this pattern, no other backends than TVM can afford this pattern
      min_cost = 100000 # sys.maxsize
      # Even if it's an autotvm, it will lower to TVM (no-tuning) ops cuz we will build without AutoTVM logs
      # We name it as autotvm because current lowering pipeline does not differentiate between TVM and AutoTVM.
      cheapest_op = f'autotvm-{pattern.get_name()}' # fallback op name
  else:
    # Exceptional cases to block
    # - 1) tensorrt_0-Op(add)[*, *] - If the add is the sole operator, then TensorRT has an issue to execute it
    if cheapest_op == 'tensorrt_0-Op(add)[*, *]' and len(expr.checked_type.shape) != 4:
      cheapest_op = f'tvm-{pattern.get_name()}' # fallback op name

  if min_cost == float('inf') and not need_tvm_fallback_ops:
    raise Exception("No corresponding backend operators / or backend op errors out (e.g., CuDNN conv_bias_relu)")

  return cheapest_op, min_cost




