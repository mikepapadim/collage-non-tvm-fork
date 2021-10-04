import matplotlib.pyplot as plt
import pandas as pd
from plot_e2e_perf import *
from backend_perf_logger import *
from tvm.relay.transform.utility.plot_utils import set_plt_font_size
from tvm.relay.transform.backend_operator.target import *
from scipy import stats
import argparse

import sys

def draw_e2e_perf_plot_normalized(df, args, is_diff_batch=False):
    df.plot.bar(figsize=(24, 5), width=0.7)

    # Save figures
    plt.xlabel("")
    plt.ylabel('Normalized Performance')
    # plt.ylabel('Inference Time (ms)')

    plt.grid(axis='y', zorder=-2.0)
    plt.xticks(rotation=0)
    plt.legend(ncol=args.n_method, loc='upper center', bbox_to_anchor=(0.48, 1.2), handletextpad=0.3, borderpad=0.3, labelspacing=0.15)
    plt.savefig(f"{EXP_RESULT_PATH}/plots/{args.plot_name}_perf_norm_{args.hw}_{args.batch_size}.png", bbox_inches='tight')

if __name__ == "__main__":
    set_plt_font_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("-hw", "--hw", help="target hardware")
    parser.add_argument("-bs", "--batch-size", default=1, type=int, help="batch size")
    args = parser.parse_args()

    print(args)

    df = pd.read_csv(DP_BACKEND_PERF_LOG_PATH)
    df = df[(df['hw'] == args.hw) & (df['batch_size'] == args.batch_size)]
    df = df.drop(columns=['hw', 'batch_size', 'std_perf'])
    df = df.set_index('network')
    df = df.pivot_table(values='mean_perf', index=df.index, columns='backends', aggfunc='first')

    # Plot adding one backend at a time
    backends_to_print = ['cublas','cublas-cudnn','cublas-cudnn-tensorrt', 'cublas-cudnn-tensorrt-autotvm']
    backend_rename_dic = {'cublas':'Collage (1 Backend)','cublas-cudnn':'Collage (2 Backends)',
                          'cublas-cudnn-tensorrt':'Collage (3 Backends)','cublas-cudnn-tensorrt-autotvm':'Collage (4 Backends)'}
    best_backend = 'Collage (4 Backends)'
    args.n_method = 4
    args.plot_name = 'backend_inc'

    # Plot all combinations of three backends performance
    #backends_to_print = ['cudnn-tensorrt-autotvm','cublas-tensorrt-autotvm',
    #                     'cublas-cudnn-autotvm', 'cublas-cudnn-tensorrt']
    #backend_rename_dic = {'cudnn-tensorrt-autotvm':'-cuBLAS','cublas-tensorrt-autotvm':'-cuDNN',
    #                      'cublas-cudnn-autotvm':'-TensorRT','cublas-cudnn-tensorrt':'-TVM'}
    #best_backend = '-TVM'
    #args.n_method = 4
    #args.plot_name = 'backend_dec'

    df = df[backends_to_print]
    df = df.rename(columns=backend_rename_dic)
    df = df.rename(index=NET_NAME_TO_OFFICIAL)

    # Normalize perf
    for method in df:
        df[method] = df[best_backend] / df[method]

    print(df)

    # Add Geomean
    df.loc['GeoMean'] = stats.gmean(df.iloc[0:5, :], axis=0)
    draw_e2e_perf_plot_normalized(df, args)
