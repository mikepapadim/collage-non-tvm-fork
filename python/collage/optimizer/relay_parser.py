from tvm.relay.dataflow_pattern import *
from tvm import relay
import tvm.relay.testing as testing
import tvm.contrib.graph_executor as runtime
import numpy as np

from ..pattern_manager.pattern import relayop_to_varnames
from ..pattern_manager.utils import get_data_shape

class TensorInfo:
    def __init__(self, name, shape, dtype='float32'):
        self.name = name
        self.type_annotation = tvm.relay.TensorType(shape, dtype)
        
        # Syntactic sugar
        self.shape = shape
    
    def get_var_args(self):
        return [self.name, self.type_annotation]
    
    def __str__():
        return f"TensorInfo({self.name}, TensorType({self.shape}, {self.dtype})"
    
"""

Build Relay operators from TENSAT IR with tensor_info(var name, shape, data type), arguments
Here are steps to build Relay operators with type inference

1. Set input_tensor_info (var_name, shape, dtype) for input variables
2. Create input variables with corresponding tensor_info
3. Set output_tensor_info to propagate shapes across e-nodes
4. Set the corresponding Relay operator to relay_expr

We don't need to set TensorType (shape, dtype) - type_args for the Relay operator.
It will be inferred automatically by the Relay runtime type system.

Warnings
- We assume that num won't be used as inputs to other relay operators.
- We assume single data type and target for a model now.
- We assume that there is always a single output from a single operator

Here are some rules to meet for valid Relay IR
- the name of Var should be "data", "weight", and any predefined names.
""" 
class RelayParser:
    def __init__(self, target, dtype='float32'):
        self._op_to_parse_func = {
            "Num": self.parse_num,
            "Var": self.parse_var,
#             "Input": self.parse_input,
#             "Weight": self.parse_input,
            "Conv2d": self.parse_conv2d,
            "Ewadd": self.parse_ewadd,
            "Relu":self.parse_relu,
        }
        
        # Only Relu and None type are supported for Relay IR
        self._taso_id_to_activation = {
            0: None,
#             1: "sigmoid",
            2: relay.op.nn.relu#"relu",
#             3: "tanh",
        }
        self._dtype = dtype
        self._target = target

    def convert_to_relay_ir(self, e_node, e_graph):
        assert e_node._op_name in self._op_to_parse_func
        
        parse_func = self._op_to_parse_func[e_node._op_name]
        parse_func(e_node, e_graph)

    def create_vars(self, relay_op_name, e_node):
        var_arr = []
        
        # We don't need to use var_names at this point
        var_names = relayop_to_varnames[relay_op_name]
        assert len(e_node.input_tensor_info) == len(var_names)
        
        for tensor_info in e_node.input_tensor_info:
            var_arr.append(relay.var(*tensor_info.get_var_args()))
        
        return var_arr
    
    # Helper function to reduce the code volume
    def get_first_e_node_from_e_class(self, e_class, e_graph):
        assert len(e_graph._e_class2e_node_id[e_class]) > 0
        
        e_node_id = e_graph._e_class2e_node_id[e_class][0]
        return e_graph._id2e_node[e_node_id]

    def get_num_from_e_class(self, e_class, e_graph):
        e_node = self.get_first_e_node_from_e_class(e_class, e_graph)
        assert e_node._op_name == 'Num'
        return e_node._op_args['num']
    
    def assign_input_tensor_info(self, e_node, e_graph):
        # 1. Set input_shape for input variables
        # Warning: We assume children are in the same order of relay.op.args
        for idx, child_e_class in enumerate(e_node.children):
            # Pick any e-node to get output shape
            child_e_node = self.get_first_e_node_from_e_class(child_e_class, e_graph)
            e_node.input_tensor_info.append(child_e_node.output_tensor_info)
            # Ideally, we want to have assertion to check if
            # children are in the same order of relay.op.args
#             assert var_names[idx] == child_e_node
            
    # 3. Set output_shape to propagate shapes across e-nodes
    # Delegate output shape inference to the relay runtime type system 
    def assign_output_tensor_info(self, e_node, e_graph, out_var_name):
        expr = e_node.relay_expr
        target_str = self._target.__str__() # e.g., "cuda"

        # Create workload
        inputs = relay.analysis.free_vars(expr)
        expr_func = relay.Function(inputs, expr)
        net, params = testing.create_workload(expr_func)

        # Build the subgraph
        ctx = tvm.device(target_str, 0)
        lib = relay.build_module.build(net, target_str, params=params)
        module = runtime.GraphModule(lib["default"](ctx))

        # Setup execution
        data_shape = get_data_shape(expr)
        data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
        module.set_input("data", data)

        # Execute the subgraph
        module.run()

        # Get output
        assert module.get_num_outputs() == 1
        out = module.get_output(0, None)

        e_node.output_tensor_info = TensorInfo(out_var_name, out.shape, self._dtype)
        
    def parse_relu(self, e_node, e_graph):
        # 1. Set input_shape and output_shape
        assert len(e_node.children) == 1
        self.assign_input_tensor_info(e_node, e_graph)
        relay_vars = self.create_vars("nn.relu", e_node)
        e_node.relay_expr = relay.op.nn.relu(*relay_vars)
        
        self.assign_output_tensor_info(e_node, e_graph, 'data')
        
        
    def parse_ewadd(self, e_node, e_graph):        
        assert len(e_node.children) == 2
        self.assign_input_tensor_info(e_node, e_graph)
        assert e_node.input_tensor_info[0].shape == e_node.input_tensor_info[1].shape
        
        relay_vars = self.create_vars("add", e_node)
        e_node.relay_expr = relay.op.add(*relay_vars)
        
        self.assign_output_tensor_info(e_node, e_graph, 'data')

        
    # Note that convolution operator have the attribute of activation
    # If the activation is not None, we need one more activation Relay IR
    # to represent this e-node
    def parse_conv2d(self, e_node, e_graph):
        # padding mode of 0 and 1 mean same and valid option of TASO
        # Link: https://www.pico.net/kb/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-tensorflow
        def get_padding(padding_mode, sh, sw, kh, kw, ih, iw):
            # Note that Relay only supports odd kernel size for padding of the SAME option
            assert kh % 2 == 1 and kw % 2 == 1
            
            padding = None
            if padding_mode == 0: # SAME
                oh = np.ceil(float(ih)/sh)
                ow = np.ceil(float(iw)/sw)
                pad_along_h = max((oh - 1)*sh + kh - ih, 0)
                pad_along_w = max((ow - 1)*sw + kw - iw, 0)
                assert int(pad_along_h) % 2 == 0 and int(pad_along_w) % 2 == 0 # if kernel size is odd
                padding = (int(pad_along_h/2), int(pad_along_w/2))
            elif padding_mode == 1: # VALID
                padding = (0, 0)
            else:
                # Padding mode should be either 0 or 1
                assert False
            
            return padding
        
            
        self.assign_input_tensor_info(e_node, e_graph)
        
        # Get operator arguments
        stride_h = self.get_num_from_e_class(e_node._op_args["stride_h_eclass_id"], e_graph)
        stride_w = self.get_num_from_e_class(e_node._op_args["stride_w_eclass_id"], e_graph)
                
        padding_mode = self.get_num_from_e_class(e_node._op_args["padding_eclass_id"], e_graph)
        activation_id = self.get_num_from_e_class(e_node._op_args["activation_eclass_id"], e_graph)
        
        strides = (stride_h, stride_w)
        
        # Assume kernel data layout of OIHW
        assert e_node.input_tensor_info[0].name == 'data' and e_node.input_tensor_info[1].name == 'weight'
        oc = e_node.input_tensor_info[1].shape[0]
        kernels = e_node.input_tensor_info[1].shape[2:]
        ih, iw = e_node.input_tensor_info[0].shape[2:]
        padding = get_padding(padding_mode, *strides, *kernels, ih, iw)
                    
        relay_vars = self.create_vars("nn.conv2d", e_node)
        e_node.relay_expr = relay.op.nn.conv2d(*relay_vars, strides=strides, padding=padding, 
                                               channels=oc, kernel_size=kernels)
        
        # Apply activation layer
        # Currently, only Relu is supported in Relay
#         assert activation_id in self._taso_id_to_activation
#         if self._taso_id_to_activation[activation_id] is not None:
#             e_node.relay_expr = self._taso_id_to_activation[activation_id](e_node.relay_expr)
        
        self.assign_output_tensor_info(e_node, e_graph, 'data')
        

    def parse_var(self, e_node, e_graph):
        var_name, shape = e_node._op_args["var_name"], e_node._op_args["shape"]
        e_node.output_tensor_info = TensorInfo(var_name, shape, self._dtype)
        e_node.relay_expr = relay.var(*e_node.output_tensor_info.get_var_args())
        
    def parse_input(self, e_node, e_graph):
        # There should be no input node after preprocessing
        assert False

    def parse_num(self, e_node, e_graph):
        num = e_node._op_args["num"]
        
        # We assume that num won't be used as inputs to other relay operators
        e_node.output_tensor_info = None
        e_node.relay_expr = relay.Constant(tvm.nd.array(np.array([num], dtype=self._dtype)))
