from .op_config import MeasuredConfigs
import os
import matplotlib.pyplot as plt
import pandas as pd

import pickle
from .op_type import OpType
from .op_config import Config
# from target import Target

fw, fh = 15, 4

def gen_op_key(key):
    return f"{key._op_type}, {key._data_shape}, {key._attrs}"

def set_plt_font_size():
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22

    plt.style.use('seaborn-paper')
    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('axes', linewidth=2)
    plt.rc('lines', linewidth=3)

def get_processed_dp(measured_configs, target_batch_size):
    op2target_perf = {}
    # Ignore std of perf
    for config, (perf, _) in measured_configs.measured_configs.items():
        op_key = gen_op_key(config)
        # if config._data_shape[0] != target_batch_size:
        #     continue
        target_name = config._op_name.split("_")[0]
        if op_key in op2target_perf:
            if target_name not in op2target_perf:
                op2target_perf[op_key][target_name] = perf
        else:
            op2target_perf[op_key] = {target_name: perf}

    # Set new op name with ID
    op2id = {}
    new_op2target_perf = {}
    for key, val in op2target_perf.items():
        op_name = key.split(",")[0]
        if op_name not in op2id:
            op2id[op_name] = 1
        else:
            op2id[op_name]+= 1
        op_name = f"{op_name}_{op2id[op_name]}"

        # Check why auto-tuned TVM_GPU operators fall behind a lot
    #     if op_name in ['conv2d_4', 'conv2d_7', 'conv2d_10']:
    #         print(key)
        new_op2target_perf[op_name] = val

    return pd.DataFrame(new_op2target_perf).T

def filter_df_with_regex(df, regex, cols_to_exclude):
    filtered_df = df.filter(regex=regex, axis=0)
    filtered_df = filtered_df[filtered_df.columns.difference(cols_to_exclude)]

    return filtered_df

def draw_plot(df, fig_name):
    df.plot.bar(figsize=(fw, fh))

    x_label_invisible = False

    # Save figures
    plt.xlabel('Operators')
    plt.ylabel('Inference Time')

    this_code_path = os.path.dirname(os.path.abspath(__file__))
    fig_name = f'{this_code_path}/../results/plots/{fig_name}'
    if x_label_invisible:
        ax1 = plt.axes()
        x_axis = ax1.axes.get_xaxis()
        x_axis.set_visible(False)

    plt.xticks(rotation=45)
    plt.savefig(fig_name, bbox_inches='tight')

def plot_resnet50():
    # Conv GPU plots
    conv_df = filter_df_with_regex(df=df, regex = 'conv2d_\d{1,2}',
                                   cols_to_exclude=['cublas'])#, 'cudnn', 'tensorrt'])#'tvmgpu-no-tuning'])
    draw_plot(df=conv_df, fig_name=f'rtx_{network_name}_bn{target_batch_size}_conv.png')

    # # Fused ops
    fused_df = df.filter(like='+', axis=0)
    fused_df = fused_df[fused_df.columns.difference(['cudnn'])]
    draw_plot(df=fused_df, fig_name=f'rtx_{network_name}_bn{target_batch_size}_fused.png')

    # Other ops
    # forbidden_str = ['+','conv2d','dense']
    # re_str = '|'.join(forbidden_str)
    drop_str = [f"conv2d_{i}" for i in range(1, 23)] + [val for val in df.index.values if "+" in val]# + ['dense_1']
    other_df = df.drop(index=drop_str)
    other_df = other_df[other_df.columns.difference(['cublas'])]
    draw_plot(df=other_df, fig_name=f'rtx_{network_name}_bn{target_batch_size}_others.png')

def plot_bert():
    # Matmul (Dense) GPU plot
    # batch_matmul_\d{1,2}
    # print(df[df.index.str.match('conv*')== False])
    dense_df = filter_df_with_regex(df=df, regex='(:?dense_\d{1,2}|batch_matmul_\d{1,2})',
                                    cols_to_exclude=['tvmcpu'])
    # print(dense_df)
    draw_plot(df=dense_df, fig_name=f'rtx_{network_name}_bn{target_batch_size}_matmul.png')

    # Other ops
    # forbidden_str = ['+','conv2d','dense']
    # re_str = '|'.join(forbidden_str)
    drop_str = [f"dense_{i}" for i in range(1, 3)] + [f"batch_matmul_{i}" for i in range(1, 2)]
    other_df = df.drop(index=drop_str)
    other_df = other_df[other_df.columns.difference(['tvmcpu'])]

    draw_plot(df=other_df, fig_name=f'rtx_{network_name}_bn{target_batch_size}_others.png')

if __name__ == "__main__":
    target_batch_size = 1
    network_name = 'bert'
    hw_name = 'rtx2070'

    NETWORK_TO_PLOT_FUNC = {
        'resnet50': plot_resnet50,
        'resnext50': plot_resnet50,
        'resnext50': plot_bert,
    }

    set_plt_font_size()

    measured_configs = MeasuredConfigs()
    measured_configs.load_from_log(hw_name)
    # measured_configs.measured_configs

    df = get_processed_dp(measured_configs, target_batch_size)

    # df['tensorrt'] = df['tvmgpu']/df['tensorrt']
    # df['cudnn'] = df['tvmgpu']/df['cudnn']
    # df['cublas'] = df['tvmgpu']/df['cublas']
    # # df['tvmcpu'] /= df['tvmgpu']
    # df['tvmgpu'] /= df['tvmgpu']

    # Conv GPU plots
    # conv_df = filter_df_with_regex(df=df, regex = 'conv2d_\d{1,2}',
    #                                cols_to_exclude=['tvmcpu', 'cublas'])#, 'cudnn', 'tensorrt'])#'tvmgpu-no-tuning'])
    # draw_plot(df=conv_df, fig_name=f'rtx_{network_name}_bn{target_batch_size}_conv.png')

    # Matmul (Dense) GPU plot
    # batch_matmul_\d{1,2}
    # print(df[df.index.str.match('conv*')== False])
    dense_df = filter_df_with_regex(df=df, regex='(:?dense_\d{1,2}|batch_matmul_\d{1,2})',
                                   cols_to_exclude=['tvmcpu'])
    # print(dense_df)
    draw_plot(df=dense_df, fig_name=f'rtx_{network_name}_bn{target_batch_size}_matmul.png')

    # Other ops
    # forbidden_str = ['+','conv2d','dense']
    # re_str = '|'.join(forbidden_str)
    drop_str = [f"dense_{i}" for i in range(1, 3)] + [f"batch_matmul_{i}" for i in range(1, 2)]
    other_df = df.drop(index=drop_str)
    other_df = other_df[other_df.columns.difference(['tvmcpu'])]

    draw_plot(df=other_df, fig_name=f'rtx_{network_name}_bn{target_batch_size}_others.png')
