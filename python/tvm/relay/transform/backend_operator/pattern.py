# currently the Pattern class does not add any additional attributes from TVM's dataflow patterns
class Pattern(object):
    def __init__(self, pattern):
        self._pattern = pattern
        
        # Serialize and reverse patterns
        
    def __eq__(self, another):
        return isinstance(another, Pattern) and self._pattern == another._pattern

    def __hash__(self):
        return hash(self._pattern)

    def __repr__(self):
        return str(self._pattern)

    def get_pattern(self):
        return self._pattern
    
#     def get_pattern_tree(self):
#         return self._pattern_tree