import pandas as pd
from tvm.relay.transform.backend_operator.target import BEST_MATCH_LOG, LOG_PATH
from tvm.relay.transform.backend_operator.plot_ops import set_plt_font_size
import os
import matplotlib.pyplot as plt

set_plt_font_size()

# net_name = "resnet50"
net_name = "resnext50_32x4d"
# net_name = "nasrnn"
file_name = f"time_perf_{net_name}"
df = pd.read_csv(f"{LOG_PATH}/{file_name}.log", index_col=0)
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