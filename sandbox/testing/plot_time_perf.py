import pandas as pd
from tvm.relay.transform.backend_operator.target import BEST_MATCH_LOG, EVAL_RESULT_LOG_PATH
from tvm.relay.transform.utility.plot_utils import set_plt_font_size

import os
import matplotlib.pyplot as plt
import sys

from plot_e2e_perf import NET_NAME_TO_OFFICIAL, FINAL_NETWORKS
from e2e_perf_logger import E2EPerfLogger
import numpy as np

def plot_single_net(net_name):
    file_name = f"time_perf_{net_name}"
    df = pd.read_csv(f"{EVAL_RESULT_LOG_PATH}/{file_name}.log", index_col=0)
    print(df)

    fw, fh = 15, 4
    # df.plot.bar(figsize=(fw, fh))
    df.plot(figsize=(fw, fh))

    x_label_invisible = False

    # Save figures
    plt.xlabel('Tuning time (secs)')
    plt.ylabel('Best perf (inference time in ms)')

    this_code_path = os.path.dirname(os.path.abspath(__file__))
    fig_name = f'{this_code_path}/../analysis/results/plots/{file_name}.png'
    if x_label_invisible:
        ax1 = plt.axes()
        x_axis = ax1.axes.get_xaxis()
        x_axis.set_visible(False)

    # plt.xticks(rotation=45)
    plt.savefig(fig_name, bbox_inches='tight')

def fill_up_missing_first_val(tuning_time, inf_time, net_df, net_name, hw, batch_size):
    def load_dp_perf(net_name, hw):
        perf_dic = E2EPerfLogger().read_dict_from_csv()
        key = E2EPerfLogger().gen_dic_key(hw, str(batch_size), net_name, 'DP')
        mean_perf, _ = perf_dic[key]

        return float(mean_perf)

    # If there is a missing first dp perf
    dp_inf_time = load_dp_perf(net_name, hw)
    if inf_time[0] < dp_inf_time:
        inf_time[0] = dp_inf_time
        print(
            f"[{net_name}] DP inf time is longer than best inf time of two-level after first iteration, measurement error")

    return tuning_time, inf_time, dp_inf_time

def plot_all_nets(networks, hw, batch_size):
    plot_file_name = f"time_perf_{hw}"

    fig_size = (10,6)
    plt.figure(figsize=fig_size)

    for net_name in networks:
        file_name = f"time_perf_{net_name}_{hw}_bs{batch_size}"
        # net_df = pd.read_csv(f"{LOG_PATH}/eval_results/rtx2070_bs1/210905/{file_name}.log", index_col=0)
        net_df = pd.read_csv(f"{EVAL_RESULT_LOG_PATH}/{hw}_bs1/{file_name}.log", index_col=0)

        tuning_time = net_df.index.tolist()
        inf_time = net_df.iloc[:, 0].tolist()

        tuning_time, inf_time, dp_inf_time = fill_up_missing_first_val(tuning_time, inf_time, net_df, net_name, hw, batch_size)
        # dp_inf_time = inf_time[0]
        tuning_time, inf_time = np.array(tuning_time) / 60.0, np.array(inf_time)

        # Cut array up to 3 hours
        cond = np.vectorize(lambda t: t < 60)
        tuning_time = tuning_time[cond(tuning_time)]
        inf_time = inf_time[:len(tuning_time)]

        rel_speed_up = dp_inf_time/inf_time

        net_name = NET_NAME_TO_OFFICIAL[net_name]
        plt.plot(tuning_time, rel_speed_up, label=net_name)

    # plt.xlabel('Tuning time (secs)')
    plt.xticks(range(10, 61, 10))
    plt.xlabel('Optimization Time (Mins)')
    plt.ylabel('Relative Speedup')
    plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.45, 1.25), labelspacing=0.3, columnspacing=1.0)

    this_code_path = os.path.dirname(os.path.abspath(__file__))
    fig_name = f'{this_code_path}/../analysis/results/plots/{plot_file_name}.png'
    plt.savefig(fig_name, bbox_inches='tight')

if __name__ == "__main__":
    #plot_single_net('resnet50')

    set_plt_font_size()
    #hw = 'rtx2070'
    hw = 'v100'
    plot_all_nets(FINAL_NETWORKS, hw, 1)
