from collage.optimizer.custom_fusion_pass import measure_end_to_end_user_defined
from workloads.torch_workloads import get_network_from_torch

import sys

assert(len(sys.argv) == 6)

net_name = sys.argv[1]
target_str = sys.argv[2]
batch_size = int(sys.argv[3])
autotvm_tuning_log = sys.argv[4]
backends = sys.argv[5].split(",")

mod, params, shape_dict, _ = get_network_from_torch(net_name, batch_size)
perf, std = measure_end_to_end_user_defined(mod["main"], params, shape_dict, target_str, net_name, batch_size, autotvm_tuning_log, backends)

print("##result: ", perf, std, file=sys.stderr)
