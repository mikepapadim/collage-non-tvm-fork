import tvm
from tvm import relay, autotvm
from collage.pattern_manager.pattern_registry import PatternRegistry
from collage.pattern_manager.utils import is_function_node
from collage.pattern_manager.cost_func import (
    NETWORK_FUNC_ATTR, HW_FUNC_ATTR, BATCH_SIZE_ATTR, BACKEND_LIST_ATTR, AUTOTVM_LOG
)

pattern_registry = PatternRegistry.get()

class Module:
    def __init__(self):
        pass

    # [TODO] Provide user-level access for backend registration.
    def add_backend_pattern(self):
        assert 0, "Need to implement"

    def add_backend_pattern_rule(self):
        assert 0, "Need to implement"

    def add_pattern_generator(self):
        assert 0, "Need to implement"

    def add_backend_codegen(self):
        assert 0, "Need to implement"

    def get_available_backends(self):
        assert 0, "Need to implement"

    def optimize_backend_placement(self, optimizer, backends, network_name, mod, params, device, target, batch_size):
        
        net = mod["main"]
        assert is_function_node(net)
        from collage.optimizer.custom_fusion_pass import CustomFusionPass
        assert(optimizer == "op-level" or optimizer == "two-level")
        optimizer = CustomFusionPass.DP if optimizer == "op-level" else CustomFusionPass.TWO_LEVEL_OPT

        net = net.with_attr("CustomFusionPass", optimizer)
        # We need to pass these for now. 
        net = net.with_attr(NETWORK_FUNC_ATTR, network_name)
        net = net.with_attr(HW_FUNC_ATTR, device)
        net = net.with_attr(BATCH_SIZE_ATTR, batch_size)
        #net = net.with_attr(BACKEND_LIST_ATTR, "1,2,3,5")

        with autotvm.apply_history_best(AUTOTVM_LOG):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(net, target, params=params)

        return lib
    