import re
import ast
from collections import defaultdict

from .relay_parser import RelayParser
from .op_arg_parser import OpArgParser

class ENode:
    def __init__(self, node_id, e_class, op_name, op_args, parse_func):
        self._node_id = node_id
        self._e_class = e_class
        self._op_name = op_name
#         self.pattern = op_dic[op_name]

        # self.children includes children e-classes of each e-node
        self.children, self._op_args = parse_func(op_args)
        self.relay_expr = None
        self.input_tensor_info = []
        # Assume that there is a single output shape (no multiple outputs)
        self.output_tensor_info = None
        
    def printit(self):
        print(self._node_id, self.children, self._op_name, self._op_args)

# Warning: We assume that the ML model has a single input
# for running DP over e-graphs
class EGraph:
    def __init__(self, target, dtype='float32'):
        self._e_class2e_node_id = defaultdict(list) 
        self._e_class2topo_order = defaultdict(None)
        self._id2e_node = defaultdict(None) 
        
        # Note that children presents e-classes
        self._arg_parser = OpArgParser()
        self._relay_parser = RelayParser(target, dtype)
#         self._type_infer_engine = TypeInferenceEngine(target, dtype)
        
        self._max_node_id = 0
        
        # Root means the final operator node of comp graph
        # We assume that root e_class has the largest index
        self._max_e_class = self._root_e_class = -1
        
        # For DP
        # Parents mean the parent e-node ids
        self._e_class2parents = defaultdict(list)
        self._input_e_class = -1
    
    def build(self, e_graph_path):
        with open(e_graph_path) as f:
            for line in f:
                line = line.strip('\n')
                assert len(line.split(":")) == 2

                e_class, e_nodes_text = int(line.split(":")[0]), line.split(":")[1][2:-1]
                self.add_e_nodes(e_class, e_nodes_text)
#                 print(e_nodes_text)
#                 egraph.parse_e_nodes(e_nodes_text, e_class)


        # print(e_graph._e_class2e_node_id)
        # e_graph.print_e_nodes()
        self.preprocess_e_graph()
        # print("-"*30)
        # print(e_graph._e_class2e_node_id)
        # e_graph.print_e_nodes()
        assert self.check_root()
        print(f"root eclass_id : {self._root_e_class}")
        print(f"input eclass_id : {self._input_e_class}")
        self.to_relay_ir(self._root_e_class, self._max_e_class)
        
        # Assign parent e-nodes to e-class
        self._assign_parents_to_e_class(self._root_e_class)
        
    def _assign_parents_to_e_class(self, e_class_id):
        e_node_ids = self._e_class2e_node_id[e_class_id]
        for e_node_id in e_node_ids:
            e_node = self._id2e_node[e_node_id]
            children_eclasses = e_node.children
            for child_e_class in children_eclasses:
                self._e_class2parents[child_e_class].append(e_node_id)
                self._assign_parents_to_e_class(child_e_class)

    def check_root(self):
        # Assume that there is no cycle
        is_root = True
        for cur_node_id, cur_node in self._id2e_node.items():
            if self._root_e_class in cur_node.children:
                is_root = False
                break
        return is_root
        
    def add_e_nodes(self, e_class, e_nodes_text):
        # parse_e_nodes returns array of e_nodes)
        self._e_class2e_node_id[e_class] += self.parse_e_nodes(e_class, e_nodes_text)
        self._max_e_class = self._root_e_class = e_class if self._root_e_class < e_class else self._root_e_class
    
    def parse_e_nodes(self, e_class, e_nodes_text):
#         e_nodes_text = "Conv2d([0, 0, 1, 2, 4, 6]),Conv2d([0, 0, 1, 2, 4, 6])"
        e_nodes_text = re.split("\(*\)", e_nodes_text)[:-1]
        e_nodes = []
        for node_text in e_nodes_text:
            # ast.literal_eval guarantee 
            op_name, op_args = node_text.split("(")[0].split(",")[-1], ast.literal_eval(node_text.split("(")[1])
            op_args = op_args if type(op_args) == list else [op_args]
            
            node_id = self._max_node_id
            self._max_node_id += 1
            
            self._id2e_node[node_id] = ENode(node_id, e_class, op_name, op_args[:], 
                                             self._arg_parser.parse_func(op_name))
            e_nodes.append(node_id)

#             for key, value in self._id2e_node[node_id]._op_args.items():
#                 print(f"{key} -> {type(value)}")
#             self._id2e_node[node_id].printit()
        
        return e_nodes

    # Remove redundant e-nodes with the type of "Input" or "Weight"
    # Note that there can be remaining e-node ids corresponding to removed e-node
    def preprocess_e_graph(self):
        op_name_to_relay_var_name = {'Input': "data", "Weight":"weight"}
        # Just to make sure there is only one input
        n_input = 0
        
        for cur_node_id in range(self._max_node_id):
            cur_node = self._id2e_node[cur_node_id]
            assert cur_node != None
            
            # Mark input node for DP
            if cur_node._op_name in ['Input']:
                n_input += 1
                self._input_e_class = cur_node._e_class
            
            if cur_node._op_name in ['Input', 'Weight']:
                assert len(cur_node.children) == 1
                child_id = cur_node.children[0]
                child_e_class_id = self._id2e_node[child_id]._e_class
                self._id2e_node[cur_node_id] = self._id2e_node[child_id]
                
                # Update e_class and node_id 
                self._id2e_node[cur_node_id]._e_class = cur_node._e_class
                self._id2e_node[cur_node_id]._node_id = cur_node._node_id
                
                # Register variable name for Relay IR conversion
                relay_var_name = op_name_to_relay_var_name[cur_node._op_name]
                self._id2e_node[cur_node_id]._op_args["var_name"] = relay_var_name
                
                # Remove child nodes
                self._id2e_node.pop(child_id, None)
                self._e_class2e_node_id[child_e_class_id].remove(child_id)
                if len(self._e_class2e_node_id[child_e_class_id]) == 0:
                    self._e_class2e_node_id.pop(child_e_class_id, None)
        
        # Warning: We assume that the ML model has a single input
        # for running DP over e-graphs
        assert n_input == 1
    
    # 1) Parse it to Relay IR
    # 2) Mark topological order of e-class (for DP optimizer)
    def _set_topological_order(self, e_class, topo_order):
        if (e_class not in self._e_class2topo_order or 
            self._e_class2topo_order[e_class] > topo_order):
            self._e_class2topo_order[e_class] = topo_order
                              
    def to_relay_ir(self, e_class_id, topological_order):
        e_node_ids = self._e_class2e_node_id[e_class_id]
        for e_node_id in e_node_ids:
            e_node = self._id2e_node[e_node_id]
            children_eclasses = e_node.children
            for child_e_class in children_eclasses:
                self.to_relay_ir(child_e_class, topological_order-1)
            
            # Prevent an e-node from being visited more than once
            if e_node.relay_expr == None:
                self._relay_parser.convert_to_relay_ir(e_node, self)
            
        self._set_topological_order(e_class_id, topological_order)
#                 print(f"node id : {e_node_id}")
#                 print(e_node.relay_expr)
    
    def print_e_nodes(self):
        for key, value in self._id2e_node.items():
            topo_order = -1
            if value._e_class in self._e_class2topo_order:
                topo_order = self._e_class2topo_order[value._e_class]
            print(f"(node_id, e_class, topo_order) = ({key}, {value._e_class}, {topo_order}) -> {value._op_name}, {value._op_args}")

    def print_relay_ir(self):
        for key, value in self._id2e_node.items():
            print(f"{value._op_name} -> {value.relay_expr}")