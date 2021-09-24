import matplotlib.pyplot as plt
import pandas as pd
from e2e_perf_logger import *
from tvm.relay.transform.utility.plot_utils import set_plt_font_size
from scipy import stats
import argparse

NET_NAME_TO_OFFICIAL = {'bert': 'BERT', 'nasneta':"NasNet-A", 'resnet50': 'ResNet50',
                      'resnext50_32x4d':"ResNeXt50", 'resnet50_3d':"3D-ResNet50",
                      'mobilenet_v2':"Mobilenet V2", 'nasrnn':"NasRNN", 'dcgan':'DCGAN'}

FINAL_NETWORKS = ["bert", "dcgan", "nasneta", 'resnet50_3d', "resnext50_32x4d"]

def setup_df_with_baselines_and_method(df):
    # Make sure column order is following before we normalize perf
    df = df[['cuDNN', 'AutoTVM', 'TensorRT', 'TF', 'TF-XLA', 'PyTorch', 'AutoTVM-libs', 'DP', 'Two-level']]
    df = df.rename(columns={'AutoTVM-libs': 'TVM', 'Two-level': 'Collage'})
    df = df.drop(columns=['DP'])
    df = df.loc[FINAL_NETWORKS]
    df = df.rename(index=NET_NAME_TO_OFFICIAL)
    # df = df.drop(['Mobilenet V2', 'ResNet50', 'NasRNN'])#, '3D-ResNet50', 'DCGAN', 'NasNet-A', 'ResNeXt50'])

    print(df)
    return df

def setup_df_for_normalized_perf_plot(df):
    df = setup_df_with_baselines_and_method(df)
    # Normalize the performance
    for method in df:
        print(method)
        df[method] = df['Collage'] / df[method]

    # print(df.iloc[0:5, :])

    # Add Geomean
    df.loc['GeoMean'] = stats.gmean(df.iloc[0:5, :], axis=0)

    # Correct Geomean of TF-XLA to deal with missing values
    xla_perf = [df.loc['BERT', 'TF-XLA'], df.loc['ResNeXt50', 'TF-XLA'], df.loc['NasNet-A', 'TF-XLA']]
    df.at['GeoMean', 'TF-XLA'] = stats.gmean(xla_perf, axis=0)
    print(df)

    return df

def draw_e2e_perf_plot_normalized(df, args):
    df = setup_df_for_normalized_perf_plot(df)
    df.plot.bar(figsize=(24, 5), width=0.7)

    # Save figures
    plt.xlabel("")
    plt.ylabel('Normalized Throughput')
    # plt.ylabel('Inference Time (ms)')

    plt.grid(axis='y', zorder=-2.0)
    plt.xticks(rotation=0)
    plt.legend(ncol=8, loc='upper center', bbox_to_anchor=(0.48, 1.2), handletextpad=0.3, borderpad=0.3)
    plt.savefig(f"{EXP_RESULT_PATH}/plots/e2e_perf_norm_{args.hw}_{args.batch_size}.png", bbox_inches='tight')

def draw_e2e_perf_plot_ms(df, args):
    df = setup_df_with_baselines_and_method(df)
    df.plot.bar(figsize=(24, 5), width=0.7)

    # Save figures
    plt.xlabel("")
    plt.ylabel('Inference Time (ms)')

    plt.grid(axis='y', zorder=-2.0)
    plt.xticks(rotation=0)
    plt.legend(ncol=6, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    plt.savefig(f"{EXP_RESULT_PATH}/plots/e2e_perf_ms_{args.hw}_{args.batch_size}.png", bbox_inches='tight')


if __name__ == "__main__":
    set_plt_font_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("-hw", "--hw", help="target hardware")
    parser.add_argument("-bs", "--batch-size", default=1, type=int, help="batch size")
    args = parser.parse_args()

    df = pd.read_csv(E2E_PERF_LOG_PATH, header=None)
    df.columns = E2E_PERF_COLS
    df = df[(df['HW'] == args.hw) & (df['BatchSize'] == args.batch_size)]
    df = df.drop(columns=['HW', 'BatchSize', 'Std Perf'])
    df = df.set_index('Network')
    df = df.pivot_table(values='Mean Perf', index=df.index, columns='Method', aggfunc='first')

    draw_e2e_perf_plot_normalized(df, args)
    # draw_e2e_perf_plot_ms(df, args)

# This plots the speedup breakdown of op-level and two-level
# e.g., 1.2x by op-level, 1.4x by two-level. Then 50% by op-level, 50% by graph-level
# However, this has a problem if the best baseline is better than ours.
# So no longer considered
# def draw_first_vs_second_level_plot(df):
#     # Make sure column order is following before we normalize perf
#     df = df[['PyTorch', 'AutoTVM-libs', 'cuDNN', 'AutoTVM', 'TensorRT', 'DP', 'Two-level']]
#     df = df.rename(columns={'AutoTVM-libs': 'TVM', 'Two-level': 'Graph-level', 'DP':'Op-level'})
#     df = df.drop(['Mobilenet V2', 'ResNet50', 'NasRNN'])
#
#     df_ours = df[['Op-level', 'Graph-level']]
#     df_baseline = df.drop(columns=['Op-level', 'Graph-level'])
#     best_baseline_time = df_baseline.min(axis=1)
#
#     # Normalize the performance
#     for method in df_ours:
#         df_ours[method] = best_baseline_time / df_ours[method]
#
#     print(df_ours)


# draw_first_vs_second_level_plot(df)

# df = setup_df_for_perf_inf_engine_vs_op_lib_plot(df)
# print(df)
# draw_e2e_perf_inf_engine_vs_op_lib_plot(df)

# def setup_df_for_perf_inf_engine_vs_op_lib_plot(df):
#     # [Temporary] Drop two-level and DP perf
#     print(df)
#     df = df.drop(columns=['Two-level', 'DP'])
#
#     # [Figure 1] Drop indices other than Mobilenet V2 (AutoTVM best), ResNext(AutoTVM best), and ResNet-50 (TensorRT best)
#     df = df.drop(['BERT', 'NasNet-A', '3D-ResNet50', 'NasRNN'])
#
#     df = df[['cuDNN', 'AutoTVM', 'TensorRT']]
#     df = df.reindex(index = ['ResNet50','ResNeXT50','Mobilenet V2'])
#     for method in df:
#         print(method)
#         df[method] = df['TensorRT'] / df[method]
#
#     return df
#
# def draw_e2e_perf_inf_engine_vs_op_lib_plot(df):
#     df.plot.bar(figsize=(12, 5), width=0.7)
#
#     # x_label_invisible = False
#     #
#     # if x_label_invisible:
#     #     ax1 = plt.axes()
#     #     x_axis = ax1.axes.get_xaxis()
#     #     x_axis.set_visible(False)
#
#     # Save figures
#     plt.xlabel("")
#     plt.ylabel('Normalized Performance')
#     # plt.ylabel('Inference Time (ms)')
#
#     plt.xticks(rotation=0)
#     plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.2))
#     plt.savefig(f"{EXP_RESULT_PATH}/plots/e2e_perf_inf_eng_vs_op_lib.png", bbox_inches='tight')


