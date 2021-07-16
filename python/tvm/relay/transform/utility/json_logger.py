import json
import datetime
import os

class JSONLogger:
    def read(self, file_path):
        with open(file_path, ) as f:
            dic = json.load(f)
        return dic


    def dump(self, dic, file_path):
        with open(file_path, "w") as outfile:
            json.dump(dic, outfile)

def dump_autotvm_tuning_info(tuning_option, search_time, hw_name):
    del tuning_option['measure_option']
    tuning_option["search_time"] = f"{search_time:.2f}s"
    date_now = datetime.datetime.now()
    tuning_option["cur_date"] = date_now.strftime("%m/%d/%Y, %H:%M:%S")

    this_code_path = os.path.dirname(os.path.abspath(__file__))
    net_name = tuning_option["network"]
    date_now = date_now.strftime("%m-%d:%H")
    file_path = f"{this_code_path}/../logs/json_logs/autotvm_tuning_log_{hw_name}_{net_name}_{date_now}"
    JSONLogger().dump(tuning_option, file_path)


