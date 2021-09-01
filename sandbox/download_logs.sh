#!/bin/bash

# Download AutoTVM autotuning results
scp cf:/home/byungsoj/tvm/python/tvm/relay/transform/logs/autotvm_ops_rtx2070.json /Users/bsjeon/Dropbox/research_phd/codes/tvm/python/tvm/relay/transform/logs/autotvm_ops_rtx2070.json

# Download end-to-end perf results
scp cf:/home/byungsoj/tvm/sandbox/analysis/results/e2e_perf.csv /Users/bsjeon/Dropbox/research_phd/codes/tvm/sandbox/analysis/results/e2e_perf.csv

# Download dp-tuning-time results
scp cf:/home/byungsoj/tvm/sandbox/analysis/results/dp_tuning_time.csv /Users/bsjeon/Dropbox/research_phd/codes/tvm/sandbox/analysis/results/dp_tuning_time.csv