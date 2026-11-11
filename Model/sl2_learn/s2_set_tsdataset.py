# 为每个监测变量匹配输入特征并构建 TFT 时间序列数据集。
# Match input features and build TFT time-series datasets for each monitored variable.


import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from global_config import config


class CreateTimeSeriesSet:
    @classmethod
    def create_datasets(cls, df_list, var_group_list):

        # 打印流程提示
        # Print process description
        print("\033[1;93m\n--- Process 5 ---  Creating a Time Series Set  ---\033[0m")

        # 根据［wind_farm_id］在变量信息表［feature_info.csv］中匹配监测变量
        # Match monitoring variables in the feature_info.csv according to wind_farm_id
        info = pd.read_csv("../Archive/feature_info.csv")
        required_columns = [
            "main_mv",
            "subsidiary_mv_1",
            "subsidiary_mv_2",
            "subsidiary_mv_3",
        ]
        row = info[info["wind_farm"] == config["wind_farm_id"]]
        monitoring_variables = row.iloc[0][required_columns].tolist()

        # 设置编码器和解码器的长度
        # Set the encoder and decoder length
        max_encoder_length = config["max_encoder_length"]
        if config["min_encoder_length"] == -1:
            min_encoder_length = max_encoder_length // 2
        else:
            min_encoder_length = config["min_encoder_length"]
        decoder_length = config["decoder_length"]

        # 设置是否增加额外特征
        # Set whether to add additional features
        add_relative_time_idx = config["add_relative_time_idx"]
        add_encoder_length = config["add_encoder_length"]

        # 传入训练集和测试集df
        # Import the training set and test set df
        train_df = df_list[0]
        test_df = df_list[1]

        # 设置验证集向前扩展尺寸
        # Set the validation set forward expansion size
        if config["posterior_size"] == -1:
            posterior_size = len(test_df) // 2
        else:
            posterior_size = config["posterior_size"]

        # 构建空字典用来存储每个监测变量的时间序列数据集
        # Set an empty dictionary to store the time series dataset of each monitoring variable
        datasets = {}

        # 构建空列表用来存储每个监测变量的数据集信息
        # Set an empty list to store the dataset info of each monitoring variable
        result_list = []

        # 遍历每一个监测变量
        # Traverse each monitoring variable
        for monitoring_variable in monitoring_variables:

            # 匹配：监测变量－输出标签
            # Match: Monitoring variables - Output labels
            if monitoring_variable in train_df.columns:
                monitoring_variable_avg = monitoring_variable
            elif monitoring_variable + "_avg" in train_df.columns:
                monitoring_variable_avg = monitoring_variable + "_avg"
            else:
                raise ValueError("Error, monitoring variable not found!")

            # 匹配：监测变量－输入传感器
            # Match: Monitoring variables - Input sensors
            monitoring_group = None
            for element in var_group_list:
                if element[0] == monitoring_variable:
                    monitoring_group = element[1]
                    break
            if monitoring_group is not None:
                input_sensors = [element[0] for element in var_group_list if element[1] != monitoring_group]
            else:
                raise ValueError("Error, input features not found!")

            # 匹配：输入传感器－输入特征
            # Match: Input sensors - Input features
            input_features_set = set()
            for sensor in input_sensors:
                matching_columns = [col for col in train_df.columns if col == sensor or col.startswith(sensor + "_")]
                input_features_set.update(matching_columns)
            input_features = list(input_features_set)

            # 配置时间序列数据集的参数
            # Common configuration for TimeSeriesDataSet
            dataset_config = {
                "time_idx": "time_idx",
                "target": monitoring_variable_avg,
                "group_ids": ["asset_id"],
                "min_encoder_length": min_encoder_length,
                "max_encoder_length": max_encoder_length,
                "max_prediction_length": decoder_length,
                "static_categoricals": ["asset_id"],
                "time_varying_known_reals": [
                    "time_idx",
                    "hour_sin",
                    "hour_cos",
                    "day_of_week_sin",
                    "day_of_week_cos",
                    "month_sin",
                    "month_cos",
                ]
                + input_features,
                "time_varying_unknown_reals": [monitoring_variable_avg],
                "allow_missing_timesteps": True,
                "add_relative_time_idx": add_relative_time_idx,
                "add_encoder_length": add_encoder_length,
            }

            # 构建用于训练的时间序列数据集
            # Create a TimeSeriesDataSet for training
            training = TimeSeriesDataSet(train_df, **dataset_config)

            # 将训练数据框的尾部数据合并到测试数据框
            # Merge the tail data of the training df into the test df
            context_df = train_df.groupby("asset_id").tail(max_encoder_length + posterior_size)
            combined_test_df = pd.concat([context_df, test_df], ignore_index=True)
            combined_test_df = combined_test_df.fillna(0)

            # 构建用于验证的时间序列数据集
            # Create a TimeSeriesDataSet for validation
            validation = TimeSeriesDataSet(combined_test_df, **dataset_config)

            # 向监测变量数据集信息列表中添加信息
            # Add info to the monitoring variable dataset info list
            known_reals = training.time_varying_known_reals or []
            feature_info = (
                f"{known_reals[20]}, {known_reals[21]}, etc." if len(known_reals) >= 22 else ", ".join(known_reals)
            )
            result_list.append(
                {
                    "MV": monitoring_variable,
                    "TrainLen": str(len(training)),
                    "ValidLen": str(len(validation)),
                    "FeatuLen": len(known_reals),
                    "FeatInfo": feature_info,
                }
            )

            # 向监测变量数据集字典中添加对应的时间序列训练集和验证集
            # Add the training and validation time series set to the monitoring variable dataset dictionary
            datasets[monitoring_variable] = {
                "training": training,
                "validation": validation,
            }

        # 以数据框形式打印时间序列数据集信息及其尺寸
        # Print the time-series dataset information and dimensions as a DataFrame
        result_df = pd.DataFrame(result_list)
        print("\nTime Series Set Info:")
        print(result_df.to_string(index=False))
        print(f"[{result_df.shape[0]} rows x {result_df.shape[1]} columns]")

        # 返回存放着每个监测变量对应的时间序列数据集的字典
        # Returns dictionary containing the time series dataset corresponding to each monitoring variable
        return datasets
