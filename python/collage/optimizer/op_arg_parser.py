"""
Make sure the names and order of arguments match var names in Relay IR.
e.g., ['data', "weight"] for conv2d

relayop_to_varnames = {
  "add" : ["data", "data"],
  "nn.conv2d" : ["data", "weight"],
  "nn.batch_norm" : ["data", "bn_data_gamma", "bn_data_beta", "bn_data_moving_mean", "bn_data_moving_var"],
  "nn.relu" : ["data"],
  "nn.softmax" : ["data"],
  "nn.bias_add" : ["data", "bias"],
  "nn.dense" : ["data", "weight"],
  "nn.batch_flatten" : ["data"],
  "nn.global_avg_pool2d" : ["data"],
  "nn.max_pool2d" : ["data"],
}

"""
class OpArgParser:
    def __init__(self):
        self._op_to_parse_func = {
            "Num": self.parse_num,
            "Var": self.parse_var,
            "Input": self.parse_input,
            "Weight": self.parse_input,
            "Conv2d": self.parse_conv2d,
            "Ewadd": self.parse_ewadd,
            "Relu":self.parse_relu,
        }
        
    def parse_func(self, pattern):
        assert pattern in self._op_to_parse_func
        
        return self._op_to_parse_func[pattern]

    def parse_relu(self, op_args):
        return [op_args[0]], {
            "input_eclass_id_0": op_args[0]
        }

    
    def parse_ewadd(self, op_args):
        return [op_args[0], op_args[1]], {
            "input_eclass_id_0": op_args[0],
            "input_eclass_id_1": op_args[1],
        }

    def parse_conv2d(self, op_args):
        return [op_args[4], op_args[5]], {
            "stride_h_eclass_id": op_args[0],
            "stride_w_eclass_id": op_args[1],
            "padding_eclass_id": op_args[2],
            "activation_eclass_id": op_args[3],
            "input_eclass_id_0": op_args[4],
            "weight_eclass_id_0": op_args[5],
        }

    def parse_var(self, op_args):
        op_args = op_args[0].split("@")
        var_name, op_args = op_args[0], op_args[1].split("_")
        shape = list(map(int, op_args)) 
        
        return [], {"var_name": var_name, "shape" : shape}

    def parse_input(self, op_args):
        return [op_args[0]], {"eclass_id" : op_args[0]}

    def parse_num(self, op_args):
        return [], {"num" : op_args[0]}
