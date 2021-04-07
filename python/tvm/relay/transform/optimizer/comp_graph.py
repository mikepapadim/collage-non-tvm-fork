import tvm
from tvm import relay
from ..backend_operator.utils import is_call_node, is_tuplegetitem_node, is_var_node, is_constant_node, is_function_node

class Node:
    def __init__(self, relay_expr, topological_order):
        self._relay_expr = relay_expr
        self._topological_order = topological_order
        self._hash_val = hash(relay_expr)
        self._parents = []
        self.matched_expr = {}
    
    def get_n_matched(self):
        return len(self.matched_expr.keys())

    def add_parent(self, parent_hash):
        self._parents.append(parent_hash)
        
    def get_n_parents(self):
        return len(self._parents)
    
    def get_parents(self):
        return self._parents
        
    def get_relay_expr(self):
        return self._relay_expr
    
    def __gt__(self, other):
        return self._topological_order > other._topological_order
    
    def __hash__(self):
        return self._hash_val

class ComputationGraph:
    def __init__(self, relay_expr):
        self._relay_expr = relay_expr
     
        self._memo = {}
        self._n_nodes = self._get_n_nodes(relay_expr)
        
        self._nodes = []
        self.expr2node = {}
        self._memo = {}
        self._expr2graph(relay_expr=relay_expr, topological_order=0, parent_expr=None)
        self._nodes.sort(key=lambda x: x._topological_order)

        assert self._n_nodes == len(self._nodes)
    
    def reset(self):
        for node in self._nodes:
            node.matched_expr = {}
        
    def get_relay_expr(self):
        return self._relay_expr
    
    def get_n_nodes(self):
        return self._n_nodes
        
    def _get_n_nodes(self, relay_expr):
        self._memo[hash(relay_expr)] = True
        n_nodes = 1
        
        if is_constant_node(relay_expr) or (is_var_node(relay_expr) and relay_expr.name_hint != 'data'):
            n_nodes = 0
        elif is_var_node(relay_expr) and relay_expr.name_hint == 'data':
            n_nodes = 1
        elif is_tuplegetitem_node(relay_expr):
            next_expr = relay_expr.tuple_value
            if hash(next_expr) not in self._memo:
                n_nodes += self._get_n_nodes(next_expr)
        elif is_call_node(relay_expr):
            for node_idx, node in enumerate(relay_expr.args):
                if hash(node) not in self._memo:
                    # memorize this visit to prevent it from visiting twice
                    # +1 here means counting the current node
                    n_nodes += self._get_n_nodes(node)
        else:
            raise Exception("Unexpected Relay expr type")

        return n_nodes
   
    def get_root(self):
        return self._nodes[0]

    def _add_node(self, relay_expr, topological_order, parent_expr):
        self._nodes.append(Node(relay_expr, topological_order))
        self._nodes[-1].add_parent(parent_expr)
        self.expr2node[hash(relay_expr)] = self._nodes[-1]

        self._memo[hash(relay_expr)] = True

    def _expr2graph(self, relay_expr, topological_order, parent_expr):
        if is_constant_node(relay_expr) or (is_var_node(relay_expr) and relay_expr.name_hint != 'data'):
            return
        else:
            self._add_node(relay_expr, topological_order, parent_expr)

            if is_var_node(relay_expr) and relay_expr.name_hint == 'data':
                return
            elif is_tuplegetitem_node(relay_expr):
                # If it is tuple, you should use tuple_value instead of args
                next_expr = relay_expr.tuple_value
                if hash(next_expr) not in self._memo:
                    self._expr2graph(next_expr, topological_order+1, relay_expr)
            elif is_call_node(relay_expr):
                for node_idx, node in enumerate(relay_expr.args):
                    if hash(node) not in self._memo:
                        # memorize this visit to prevent it from visiting twice
                        # +1 here means counting the current node
                        self._expr2graph(node, topological_order+1, relay_expr)
                    else:
                        # Make sure the node has a right (larger) topological order
                        # if there are multiple choices
                        if self.expr2node[hash(node)]._topological_order < topological_order:
                            self.expr2node[hash(node)]._topological_order = topological_order
                        self.expr2node[hash(node)].add_parent(relay_expr)
            else:
                raise Exception("Unexpected Relay expr type")