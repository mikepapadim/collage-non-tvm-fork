import matplotlib.pyplot as plt
import pandas as pd
from tvm.relay.transform.utility.plot_utils import set_plt_font_size
from plot_e2e_perf import NET_NAME_TO_OFFICIAL
import numpy as np

from tvm.relay.transform.pattern_manager.target import *
from tvm.relay.transform.optimizer.custom_fusion_pass import *
from collections import Counter

def draw_e2e_perf_plot(df):
    fig_size = (12, 6)
    df.plot.bar(figsize=fig_size, width=0.5, stacked=True)

    # x_label_invisible = False
    #
    # if x_label_invisible:
    #     ax1 = plt.axes()
    #     x_axis = ax1.axes.get_xaxis()
    #     x_axis.set_visible(False)

    # Save figures
    plt.xlabel("")
    plt.ylabel('Time (s)')

    plt.xticks(rotation=10)
    plt.legend(ncol=2)#, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    plt.savefig(f"{EXP_RESULT_PATH}/plots/dp_tuning_time.png", bbox_inches='tight')

# def is_valid_op(op_name):
#     if op_name.count('nn.dense')


def get_stats_from_annos(annos):
    backends = []
    ops = []
    # for anno in annos:
    #     backends.append(get_backend_from_backend_op_annotation(anno))
    #     op_name = get_op_name_from_backend_op_annotation(anno)

    backend_counter = Counter(backends)
    op_counter = Counter(ops)
    print(backend_counter)
    print(op_counter)

if __name__ == "__main__":
    set_plt_font_size()

    net_name_arr = list(NET_NAME_TO_OFFICIAL.keys())
    hw_name = 'rtx2070'
    batch_size = 1

    for net_name in net_name_arr:
        opt_info_tag = get_opt_info_tag(net_name, hw_name, batch_size)
        match_path = f"{EVAL_RESULT_LOG_PATH}/{hw_name}_bs{batch_size}/best_match_{opt_info_tag}.log"
        df = pd.read_csv(match_path)
        annos = df['annotation'].tolist()
        get_stats_from_annos(annos)
        sys.exit(0)

    # df.columns = DP_TUNING_TIME_COLS
    # df = df.drop(columns=['HW', 'Std Perf'])
    # df = df.set_index('Network')
    # df = df.pivot_table(values='Mean Perf', index=df.index, columns='Method', aggfunc='first')
    # df = df.rename(index=NET_NAME_TO_OFFICIAL)
    #
    # df = df.drop(['Mobilenet V2', 'ResNet50', 'NasRNN'])
    #
    # print(df)
    # dp_time = df['DP'].to_numpy()
    # prof_time = df['Op Profiling'].to_numpy()
    # print(prof_time / (dp_time + prof_time) * 100.0)
    # avg_percent = np.mean(prof_time / (dp_time + prof_time) * 100.0)
    # print(f"Average percentage of op profiling time is {avg_percent}")
    #
    # draw_e2e_perf_plot(df)
    #
