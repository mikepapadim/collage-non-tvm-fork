import matplotlib.pyplot as plt
import pandas as pd
from e2e_perf_logger import *
from tvm.relay.transform.utility.plot_utils import set_plt_font_size

set_plt_font_size()

def draw_e2e_perf_plot(df):
    df.plot.bar(figsize=(24, 5), width=0.7)

    # x_label_invisible = False
    #
    # if x_label_invisible:
    #     ax1 = plt.axes()
    #     x_axis = ax1.axes.get_xaxis()
    #     x_axis.set_visible(False)

    # Save figures
    plt.xlabel("")
    plt.ylabel('Normalized Performance')
    # plt.ylabel('Inference Time (ms)')

    plt.xticks(rotation=0)
    plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    plt.savefig(f"{EXP_RESULT_PATH}/plots/e2e_perf.png", bbox_inches='tight')

df = pd.read_csv(E2E_PERF_LOG_PATH, header=None)
df.columns = E2E_PERF_COLS
df = df.drop(columns=['HW', 'Std Perf'])
df = df.set_index('Network')
df = df.pivot_table(values='Mean Perf', index=df.index, columns='Method', aggfunc='first')
df = df.rename(index={'bert': 'BERT', 'nasneta':"NasNet-A", 'resnet50': 'ResNet50',
                      'resnext50_32x4d':"ResNeXT50", 'resnet50_3d':"3D-ResNet50",
                      'mobilenet_v2':"Mobilenet V2", 'nasrnn':"NasRNN"})

# Make sure column order is following before we normalize perf
df = df[['cuDNN', 'AutoTVM', 'TensorRT', 'DP']]

# Normalize the performance
for method in df:
    print(method)
    df[method] = df['DP']/df[method]

print(df)
draw_e2e_perf_plot(df)
