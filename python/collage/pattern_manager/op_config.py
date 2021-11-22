from pathlib import Path
from .utils import extract_attrs, get_input_shape
import json
import pickle
from os import path

#cur_dir_path = Path(__file__).parent.absolute()

def get_opcost_log_path(hw_name, is_readable):
  extension = "log"
  if is_readable:
    extension = "json"

  return f"operator_cost_{hw_name}.{extension}"

# @Sung: [TODO] Need to check hash conflict
# configuration includes operator name, operator type (backend operators from different targets might have the same type),
# data shape of all free variables, and node attributes
class Config(object):
  # We have data_shape and attrs as arguments for debugging purpose
  def __init__(self, op_name, pattern, expr, data_shape=None, attrs=None):
    self._op_name = op_name
    self._pattern = pattern

    if expr != None:
      # self._data_shape = tuple(get_data_shape(expr))
      # _data_shape should be shape of all the inputs not only including data variable, but also parameters (constant)
      self._data_shape = get_input_shape(expr)
      self._attrs = extract_attrs(expr)

    else:
      # Debugging purpose
      self._data_shape = data_shape
      self._attrs = attrs

  def __hash__(self):
    return hash((self._op_name, self._pattern, self._data_shape, self._attrs))

  def __eq__(self, other):
#     print(f"Check equality, {type(self._op_name)}, {type(self._pattern)}, {type(self._data_shape)}, {type(self._attrs)}")
    return (self._op_name == other._op_name and self._pattern == other._pattern
    and self._data_shape == other._data_shape and self._attrs == other._attrs)

  def __repr__(self):
    return "op_name: {0}, pattern: {1}, data_shape: {2}, attrs: {3}".format(
      self._op_name, self._pattern, self._data_shape, self._attrs)

  def __str__(self):
    return "pattern: {0}, data_shape: {1}, attrs: {2}, op_name: {3}".format(
      self._pattern, self._data_shape, self._attrs, self._op_name)



# class to save costs of already evaluated configurations so we do not need to reevaluate them
class MeasuredConfigs(object):
  def __init__(self):
    # maps configurations already measured to the measured cost (in ms)
    self.measured_configs = dict()

  def get_cost(self, config):
    if config in self.measured_configs:
      return self.measured_configs[config]
    return None

  # cost is (mean(cost), std(cost))
  def save_cost(self, config, cost):
    self.measured_configs[config] = cost

  def save_to_log(self, hw_name):
    cost_log = get_opcost_log_path(hw_name, False)
    cost_log_readable = get_opcost_log_path(hw_name, True)
    # print(f"[Save] Cost log : {cost_log}")
    with open(cost_log, 'wb+') as log:
      pickle.dump(self.measured_configs, log)

    str_configs = dict()
    for key, perf in self.measured_configs.items():
        str_configs[str(key)] = perf

    with open(cost_log_readable, 'w+') as log:
      json.dump(str_configs, log, sort_keys=True, indent=4)

  # If log doesn't exist, it uses default empty dictionary.
  def load_from_log(self, hw_name):
    cost_log = get_opcost_log_path(hw_name, False)
    # print(f"[Load] Cost log : {cost_log}")
    try:
      if path.exists(cost_log):
        with open(cost_log, 'rb') as log:
          print("Cost configurations loaded")
          self.measured_configs = pickle.load(log)
    except:
#       pass
      raise Exception(f'{cost_log} is not valid')
