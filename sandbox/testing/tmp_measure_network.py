from tvm.relay.transform.optimizer.custom_fusion_pass import measure_end_to_end_user_defined
from workloads.torch_workloads import *

import sys

assert(len(sys.argv) == 3)

net_name = sys.argv[1]
target_str = sys.argv[2]

mod, params, shape_dict, _ = get_network_from_torch(net_name, 1)
perf, std = measure_end_to_end_user_defined(mod["main"], params, shape_dict, target_str)

print("##result: ", perf, std, file=sys.stderr)
