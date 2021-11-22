from e2e_perf_logger import EXP_RESULT_PATH
import pandas as pd
import os

DP_BACKEND_PERF_LOG_PATH = f"{EXP_RESULT_PATH}/dp_backend_perf.csv"

class DPBackendPerfLogger:
    def __init__(self):
        pass

    def read_df_from_csv(self):
        if os.path.exists(DP_BACKEND_PERF_LOG_PATH):
            df = pd.read_csv(DP_BACKEND_PERF_LOG_PATH)
        else:
            df = pd.DataFrame({'hw':[], 'batch_size':[], 'network':[], 'backends':[], 'mean_perf':[], 'std_perf':[]})

        # We need to make sure 1 instead of 1.0 for key matching
        df['batch_size'] = df['batch_size'].astype(int)
        df = df.set_index(['hw', 'batch_size', 'network', 'backends'])

        return df

    def write_df_to_csv(self, df):
        df.to_csv(DP_BACKEND_PERF_LOG_PATH)

    # Note that batch_size is automatically int when loading csv
    # So we need to make sure input batch_size is also int to match a key
    def log_perf(self, hw, batch_size, network, backends, mean_perf, std_perf):
        assert isinstance(batch_size, int)

        df = self.read_df_from_csv()
        # If the index exists, it simply replace that with a new values
        df.loc[(hw, batch_size, network, backends), ['mean_perf','std_perf']] = mean_perf, std_perf
        self.write_df_to_csv(df)
