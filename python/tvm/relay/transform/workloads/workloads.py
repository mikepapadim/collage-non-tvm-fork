# Value is shape dict
WORKLOADS_DIC = {
    "resnet_block" : {"input0": [1, 64, 56, 56]},
    "resnet50" : {"input0": [1, 64, 56, 56]},
    "resnext50_32x4d" : {"input0": [1, 64, 56, 56]},
    "nasneta" : {"input0": [1, 64, 56, 56]},
    "nasrnn": {'x.1': [1, 512]},
    # "nasrnn": {'x.1': [1, 512], 'x.2': [1, 512], 'x.3': [1, 512], 'x.4': [1, 512], 'x': [1, 512]},
    "bert": {"input0": [64, 1024]},
}