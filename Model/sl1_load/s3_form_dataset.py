# 构造时间特征、标准化数据，并按全局参数划分训练集和测试集。
# Build temporal features, standardize data, and split training and test sets using global parameters.


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from global_config import config


class FormDataset:
    @classmethod
    def form_dataset(cls, df):

        # 调用全局参数
        # Calling global params
        farm_label = config["wind_farm_id"]
        train_size = config["train_size"]

        # 设置数据框和传感器表［sensor_info.csv］路径
        # Set dataframe and sensor_info.csv path
        info = pd.read_csv(f"../Archive/Wind Farm {farm_label}/comma_feature_description.csv")
        sensor_columns = info["sensor_name"].tolist()
        feature_columns = df.filter(regex="^(" + "|".join(sensor_columns) + ")").columns.tolist()

        # 转换时间格式并设置数据框的索引
        # Convert to time format and set the index of DataFrame
        df["time_stamp"] = pd.to_datetime(df["time_stamp"])
        df.set_index("time_stamp", inplace=True)
        df = df[["asset_id", "train_test", "status_type_id"] + feature_columns]

        # 提取时间周期波动特征并计算三角函数特征
        # Extracting time period features and calculate sinusoidal features
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        for column, max_val in [("hour", 24), ("day_of_week", 7), ("month", 12)]:
            df[f"{column}_sin"] = np.sin(2 * np.pi * df[column] / max_val)
            df[f"{column}_cos"] = np.cos(2 * np.pi * df[column] / max_val)

        # 删除原始时间段列并保留正弦和余弦特征
        # Drop the original period columns and retain sine and cosine features
        df = df.drop(["hour", "day_of_week", "month"], axis=1)

        # 数据标准化
        # Data Standardization
        scaler = StandardScaler()
        df[feature_columns] = scaler.fit_transform(df[feature_columns])

        # 对状态变量独热编码
        # One-hot encode the status_type
        df = pd.get_dummies(df, columns=["status_type_id"], drop_first=True)

        # 重置索引并重构时间索引
        # Reset index and create time_idx
        df_reset = df.reset_index()
        df_reset = df_reset.sort_values("time_stamp").reset_index(drop=True)
        df_reset["time_idx"] = (df_reset["time_stamp"] - df_reset["time_stamp"].min()) / pd.Timedelta(minutes=10)
        df_reset["time_idx"] = df_reset["time_idx"].astype(int)

        # 将事件［id］转换成字符串以供后续［TFT］使用
        # Convert asset_id to str for TFT model processing
        if "asset_id" in df_reset.columns:
            df_reset["asset_id"] = df_reset["asset_id"].astype(str)
        df_reset.attrs["farm_label"] = farm_label

        # 打印浏览信息（仅展示部分行列）
        # Print view (display several columns & row)
        print("\033[1;93m\n--- Process 3 ---  Forming Train and Test Sets  ---\n\033[0m")
        print("Dataset (Partial Rows and Columns):")
        print(df_reset.to_string(index=False, max_rows=6, max_cols=8))
        print(f"[{df_reset.shape[0]} rows x {df_reset.shape[1]} columns]")

        # 返回一个经处理的数据框
        # Returns a processed df
        df = df_reset

        # 根据训练集尺寸参数判断：按标签分割
        # Judging by the train_size: split by tags
        if train_size == -1:

            # 根据训练测试标识［train_test］划分数据集
            # Divide the df set according to train_test
            train_df = df[df["train_test"] == "train"].copy()
            test_df = df[df["train_test"] == "prediction"].copy()

            # 打印浏览信息（仅展示部分行列）
            # Print view (display several columns)
            print("\nTrain Set (Partial Rows and Columns):")
            print(train_df.to_string(index=False, max_rows=6, max_cols=8))
            print(f"[{train_df.shape[0]} rows x {train_df.shape[1]} columns]")
            print("\nTest Set (Partial Rows and Columns):")
            print(test_df.to_string(index=False, max_rows=6, max_cols=8))
            print(f"[{test_df.shape[0]} rows x {test_df.shape[1]} columns]")

        # 根据训练集尺寸参数判断：按尺寸分割
        # Judging by the train_size: split by size
        else:

            # 在全局参数设置的尺寸大于数据集总长度时报错
            # Display error when the train_size is greater than the length of df
            if train_size > len(df):
                raise ValueError("train_size exceed the length of the DataFrame")

            # 根据训练集尺寸［train_size］划分数据集
            # Divide the df set according to the train_size
            train_df = df.iloc[:train_size].copy()
            test_df = df.iloc[train_size:].copy()

            # 打印浏览信息（仅展示部分行列）
            # Print view (display several columns)
            print("\nTrain Set (Partial Rows and Columns):")
            print(train_df.to_string(index=False, max_rows=6, max_cols=8))
            print(f"[{train_df.shape[0]} rows x {train_df.shape[1]} columns]")
            print("\nTest Set (Partial Rows and Columns):")
            print(test_df.to_string(index=False, max_rows=6, max_cols=8))
            print(f"[{test_df.shape[0]} rows x {test_df.shape[1]} columns]")

        # 将训练集和测试集合并为一个列表
        # Combine the training set and the test set into a list
        df_list = [train_df, test_df]

        # 返回数据集列表
        # Return to the df list
        return df_list
