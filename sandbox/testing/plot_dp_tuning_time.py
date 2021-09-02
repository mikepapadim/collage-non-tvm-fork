import matplotlib.pyplot as plt
import pandas as pd
from e2e_perf_logger import *
from tvm.relay.transform.utility.plot_utils import set_plt_font_size
from plot_e2e_perf import NET_NAME_TO_OFFICIAL
import numpy as np

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

if __name__ == "__main__":
    set_plt_font_size()

    df = pd.read_csv(DP_TUNING_TIME_LOG_PATH, header=None)
    df.columns = DP_TUNING_TIME_COLS
    df = df.drop(columns=['HW', 'Std Perf'])
    df = df.set_index('Network')
    df = df.pivot_table(values='Mean Perf', index=df.index, columns='Method', aggfunc='first')
    df = df.rename(index=NET_NAME_TO_OFFICIAL)

    df = df.drop(['Mobilenet V2', 'ResNet50', 'NasRNN'])

    print(df)
    dp_time = df['DP'].to_numpy()
    prof_time = df['Op Profiling'].to_numpy()
    print(prof_time / (dp_time + prof_time) * 100.0)
    avg_percent = np.mean(prof_time / (dp_time + prof_time) * 100.0)
    print(f"Average percentage of op profiling time is {avg_percent}")

    draw_e2e_perf_plot(df)


