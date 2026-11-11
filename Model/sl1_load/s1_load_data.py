# 根据全局参数加载单个事件或合并同一风机的连续事件数据。
# Load one event or merge consecutive events from the same turbine according to global parameters.


import os
import warnings
import pandas as pd
from global_config import config


class LoadData:
    @classmethod
    def load_data(cls):

        # 调用全局参数
        # Calling global params
        wind_farm_id = config["wind_farm_id"]
        event_id = config["event_id"]
        turbine_id = config["turbine_id"]

        # 根据多事件参数的布尔值判断：处理单个事件
        # Judging by the Boolean value of multiple_comma: process a single event
        if not config["multiple_comma"]:

            # 打印流程提示和基本信息
            # Print process description and basic info
            print(
                "\033[1;93m\n--- Process 1 ---  Loading Data (Single Comma)  ---\n\033[0m"
            )
            basic_info = f"Wind Farm {wind_farm_id}, Event ID is {event_id}"
            print(basic_info)

            # 构建文件路径
            # Constructing file paths
            base_dir = "../Archive"
            farm_dir = os.path.join(base_dir, f"Wind Farm {wind_farm_id}")
            data_file = os.path.join(farm_dir, "datasets", f"comma_{event_id}.csv")
            info_file = os.path.join(farm_dir, "comma_event_info.csv")

            # 读取数据文件为数据框格式
            # Read the data file in df format
            df_s_ori = pd.read_csv(data_file)

            # 根据事件［ID］在事件信息表［comma_event_info.csv］中匹配事件信息
            # Match event info in the comma_event_info.csv according to event_id
            df_info = pd.read_csv(info_file)
            row = df_info.loc[df_info["event_id"] == event_id].iloc[0]
            event_info_s = f"\nEvent Desc: {row['event_description']}\nEvent S.Tm: {row['event_start']}\nEvent E.Tm: {row['event_end']}"
            print(event_info_s)

            # 返回一个存放完整数据的数据框
            # Return a DataFrame containing the complete data
            return df_s_ori

        # 根据多事件参数的布尔值判断：处理一个风机的所有事件
        # Judging by the Boolean value of multiple_comma: process all events of a WT
        else:

            # 打印基本信息
            # Print basic info
            print(
                "\033[1;93m\n--- Process 1 --- Loading Data (Multipal Comma) ---\n\033[0m"
            )
            basic_info = f"Wind Farm {wind_farm_id}, Turbine ID is {turbine_id}"
            print(basic_info)

            # 构建文件路径
            # Constructing file paths
            base_dir = "../Archive"
            farm_dir = os.path.join(base_dir, f"Wind Farm {wind_farm_id}")
            data_dir = os.path.join(farm_dir, "datasets")

            # 收集所有匹配的数据文件
            # Collect all matching csv files
            csv_files = [
                f
                for f in os.listdir(data_dir)
                if f.startswith("comma_") and f.endswith(".csv")
            ]

            # 收集这些数据文件对应的风机标识符
            # Collect the turbine_id corresponding to these csv files
            relevant_files = []
            for f in csv_files:
                file_path = os.path.join(data_dir, f)
                df_tmp = pd.read_csv(file_path, usecols=["asset_id"])
                if turbine_id in df_tmp["asset_id"].unique():
                    relevant_files.append(f)

            # 收集这些数据文件的时间范围
            # Collect time ranges for these csv files
            time_ranges = []
            for f in relevant_files:
                file_path = os.path.join(data_dir, f)
                df_tmp = pd.read_csv(file_path)
                df_tmp["time_stamp"] = pd.to_datetime(df_tmp["time_stamp"])
                min_t = df_tmp["time_stamp"].min()
                max_t = df_tmp["time_stamp"].max()
                time_ranges.append((f, min_t, max_t))

            # 基于时间范围判定这些数据文件中哪些是时间相连
            # Determine which of these csv files are time-connected based on the time range
            filtered_files = []
            for i, (file_a, min_a, max_a) in enumerate(time_ranges):
                overlap = False
                for j, (file_b, min_b, max_b) in enumerate(time_ranges):
                    if i == j:
                        continue
                    if min_a <= max_b + pd.Timedelta(
                        minutes=10
                    ) and max_a >= min_b - pd.Timedelta(minutes=10):
                        overlap = True
                        break
                if overlap:
                    filtered_files.append(file_a)

            # 这些数据文件的时间均不相连则触发警报
            # Warning when the time of these csv files is not connected
            if not filtered_files:
                warnings.warn("No multiple files at consecutive times!\n")
                return None

            # 加载判定为时间相连的文件
            # Load csv files determined to be time-connected
            df_all = pd.DataFrame()
            for f in filtered_files:
                file_path = os.path.join(data_dir, f)
                df_tmp = pd.read_csv(file_path)
                df_all = pd.concat([df_all, df_tmp], ignore_index=True)

            # 去重并重设时间索引
            # De-duplicate and reset time index
            df_all["time_stamp"] = pd.to_datetime(df_all["time_stamp"])
            df_all = df_all.sort_values("time_stamp").drop_duplicates(
                subset="time_stamp"
            )
            df_m_ori = df_all.reset_index(drop=True)

            # 返回一个存放完整数据的数据框
            # Return a DataFrame containing the complete data
            return df_m_ori
