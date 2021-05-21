from tvm import relay
from tvm.relay.dataflow_pattern import *
from collections import namedtuple
import numpy as np

from .pattern import Pattern

# an example of a diamond pattern that occurs in resnet-18
def get_diamond():
  relu = is_op("nn.relu")(wildcard())
  conv2d1_1 = is_op("nn.conv2d")(relu, wildcard())
  batch_norm1 = is_tuple_get_item(is_op("nn.batch_norm")(conv2d1_1, wildcard(), wildcard(), wildcard(), wildcard()), 0)
  relu1 = is_op("nn.relu")(batch_norm1)
  conv2d1_2 = is_op("nn.conv2d")(relu1, wildcard())
  conv2d2 = is_op("nn.conv2d")(relu, wildcard())
  diamond = is_op("add")(conv2d1_2, conv2d2)
  return Pattern(diamond)

# return the shape of input data to expr
def get_data_shape(expr):
  inputs = relay.analysis.free_vars(expr)
  # if is_call_node(expr):
  #   print(f"Input for expr ({expr.op}) {[inputs[0].type_annotation]}, {expr.attrs.axis}")

  # for add, shape of lhs and rhs should be identical. for all other backend ops, we take shape of "data" input arg
  # inputs[0] corresponds to Var(name_hint='data')
  # We consider two different types for that: TupleTypeNode, TensorTypeNode
  if type(inputs[0].type_annotation) == relay.TensorType:
    data_shape_imm = inputs[0].type_annotation.shape
    data_shape = list(map(lambda x: x.value, data_shape_imm))
  elif type(inputs[0].type_annotation) == relay.TupleType:
    data_shape = []
    for tup_item in inputs[0].type_annotation.fields:
      data_shape.append(tuple((map(lambda x: x.value, tup_item.shape))))
    data_shape = tuple(data_shape)
    print("data shape", data_shape)
  else:
    raise Exception(f"Unsupported Var type ({type(inputs[0].type_annotation)})")

  return data_shape

def get_data(expr):
  data_shape = get_data_shape(expr)
  if type(data_shape) == list:
    print(data_shape)
    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
  elif type(data_shape) == tuple:
    data = []
    for shape in data_shape:
      data.append(np.random.uniform(-1, 1, size=shape).astype("float32"))
      print(shape)
    data = tuple(data)
  else:
    raise Exception(f"Unsupported data shape type {type(data_shape)}")

  return data

def is_function_node(expr):
  return type(expr) == tvm.relay.Function

def is_constant_node(expr):
  return type(expr) == tvm.relay.Constant

def is_call_node(expr):
  return type(expr) == tvm.relay.expr.Call

def is_tuple_node(expr):
  return type(expr) == tvm.relay.expr.Tuple

def is_tuplegetitem_node(expr):
  return type(expr) == tvm.relay.expr.TupleGetItem

def is_call_or_tuplegetitem_node(expr):
  # If not, it means that we need to add codes to deal with other nodes
  assert is_call_node(expr) or is_tuplegetitem_node(expr) or is_var_node(expr) or is_constant_node(expr) or is_tuple_node(expr)
  return is_call_node(expr) or is_tuplegetitem_node(expr)

def is_var_node(expr):
  return type(expr) == tvm.relay.expr.Var

def no_constraints_func(config):
  return True

# given a tvm.ir.Attrs node, return list of attribute values
def get_attr_vals(expr):
  assert is_call_node(expr)
  attrs = expr.attrs 
  op_name = expr.op.name

  if attrs == None or "keys" not in dir(attrs):
    return (op_name, "")

  # print(f"{expr.op}'s Attrs : {dir(attrs)}")
  keys = attrs.keys()
  values = []
  for key in keys:
    value = attrs.get_int(key)
    v_type = type(value)

    if v_type == type(None):
      pass
    elif v_type == tvm.ir.container.Array:
      value = tuple(value)
    elif v_type == int or v_type == str or v_type == float:
      pass
    elif v_type == tvm.tir.expr.IntImm:
      value = int(value)
    elif v_type == tvm.runtime.container.String:
      value = tuple(value)
    else:
      print(key, value, v_type)
      raise Exception(f"Unexpected tvm data type ({v_type}) for attributes")
    
    values.append(value)
  
  return (op_name, tuple(zip(keys, values)))

# extract the node attributes of a relay expr. Use a list of tvm node attributes to represent each path (branch) in the expr.
# Then use an immutable set of these lists to represent all node attributes of the expr.
# Note that expr is extracted subgraph that matches backend op
def extract_attrs(expr):
  res = set()

  def helper(expr, attrs):
    if is_call_node(expr):
      attr_vals = get_attr_vals(expr)
      # print(attr_vals)
      attrs += attr_vals
      children = list(filter(is_call_or_tuplegetitem_node, expr.args))
      if len(children) == 0:
        res.add(attrs)

      for child in children:
        helper(child, attrs)

    elif is_tuplegetitem_node(expr):
      helper(expr.tuple_value, attrs)

    elif is_tuple_node(expr):
      children = list(filter(is_call_or_tuplegetitem_node, expr.fields))
      if len(children) == 0:
        res.add(attrs)

      for child in children:
        helper(child, attrs)

    elif is_var_node(expr):
      raise Exception("Should not reach var node")

    else:
      raise Exception("Expr type not implemented")

  helper(expr, ())
  return frozenset(res)

# extract the node attributes of a relay expr. Use a list of tvm node attributes to represent each path (branch) in the expr.
# Then use an immutable set of these lists to represent all node attributes of the expr. 
# def extract_attrs(expr):
#   if is_call_node(expr):
#     attrs += get_attr_vals(expr)
#     children = list(filter(is_call_or_tuplegetitem_node, expr.args))
#     if len(children) == 0:
#       return attrs

#     for child in children:
#       extract_attrs(child)

#   elif is_tuplegetitem_node(expr):
#     extract_attrs(expr.tuple_value, attrs)

#   elif is_var_node(expr):
#     raise Exception("Should not reach var node")

#   else:
#     raise Exception("Expr type not implemented")

#   helper(expr, ())
#   return res