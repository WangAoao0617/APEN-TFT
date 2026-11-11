# 根据风场和全局参数清洗功率异常值并返回清洗后的数据。
# Clean power outliers by wind farm and global parameters, then return the cleaned data.


import pandas as pd
from global_config import config


class CleanData:
    @classmethod
    def cln_data(cls, df):

        # 调用全局参数
        # Calling global params
        para_k = config["cl_std_ratio"]
        quan_bins = config["number_of_bins"]
        farm_label = config["wind_farm_id"]

        # 打印流程提示
        # Print process description
        print(
            "\033[1;93m\n--- Process 2 ---  Removing Error and Outliers  ---\n\033[0m"
        )

        # 根据风场标签判断：特殊风场（风场C）
        # Judging by the farm_label: special wind farm (Wind Farm C)
        if farm_label == "C":
            count_rule1 = 0
            count_rule2 = 0

            # 打印数据清洗结果信息
            # Print data cleaning result info
            print(f"Remove Power Abandonment Data: {count_rule1}")
            print(f"Remove Measurement Error Data: {count_rule2}")

            # 返回原始数据框
            # Return original df
            return df

        # 根据风场标签判断：一般风场（风场A或B）
        # Judging by the farm_label: normal wind farm (Wind Farm A or B)
        else:

            # 根据风场在变量信息表［feature_info.csv］中匹配功率曲线变量
            # Match power curve variables in the feature_info.csv according to farm_label
            info = pd.read_csv("../Archive/feature_info.csv")
            row = info[info["wind_farm"] == farm_label].iloc[0]

            # 根据数据清洗参数的布尔值判断：仅清洗离群数据
            # Judging by the Boolean value of data_cleaning_rules: only cleaning outliers
            if not config["data_cleaning_rules"]:

                # 规则（一）：剔除理论功率小于零的数据
                # Rule 1: Eliminate power < 0
                ws_col, power_col = (
                    row["wind_speed_feature_name"],
                    row["possible_power_feature_name"],
                )
                cond1 = df[power_col] < 0
                count_rule1 = cond1.sum()

                # 按风速等频分箱
                # Equal frequency binning by wind speed
                df = df.copy()
                df["bin"] = pd.qcut(df[ws_col], q=quan_bins, duplicates="drop")

                # 规则（二）：分箱后使用控制图方法去除离群值
                # Rule 2: Remove outliers based on the mean ± k*std after binning
                count_rule2 = 0
                for b in df["bin"].unique():
                    sub = df[df["bin"] == b]
                    mean_val = sub[power_col].mean()
                    std_val = sub[power_col].std()
                    cond2 = (sub[power_col] > mean_val + para_k * std_val) | (
                        sub[power_col] < mean_val - para_k * std_val
                    )
                    count_rule2 += cond2.sum()
                    df = df[~((df["bin"] == b) & cond2)]

                # 重置索引
                # Reset Index
                df_cleaned = df.drop(columns="bin").reset_index(drop=True)
                df_cleaned.attrs["farm_label"] = farm_label

                # 打印数据清洗结果信息
                # Print data cleaning result info
                print(f"Remove Power Abandonment Data: {count_rule1}")
                print(f"Remove Measurement Error Data: {count_rule2}")

                # 返回清洗后的数据框
                # Return cleaned df
                return df_cleaned

            else:
                ws_col, power_col = (
                    row["wind_speed_feature_name"],
                    row["gird_power_feature_name"],
                )

                # 规则（一）：剔除弃风限电数据
                # Rule 1: Remove power curtailment data
                cond1 = (df[ws_col] > 4) & (df[power_col] <= 0.005)
                count_rule1 = cond1.sum()
                df = df[~cond1]

                # 按风速等频分箱
                # Apply equal-frequency binning by wind speed
                df = df.copy()
                df["bin"] = pd.qcut(df[ws_col], q=quan_bins, duplicates="drop")

                # 规则（二）：剔除各分箱中的离群值
                # Rule 2: Remove outliers within each bin
                count_rule2 = 0

                for b in df["bin"].unique():
                    sub = df[df["bin"] == b]
                    mean_val = sub[power_col].mean()
                    std_val = sub[power_col].std()
                    cond2 = (sub[power_col] > mean_val + para_k * std_val) | (
                        sub[power_col] < mean_val - para_k * std_val
                    )
                    count_rule2 += cond2.sum()

                    # 去除异常值
                    # Remove outliers
                    df = df[~((df["bin"] == b) & cond2)]

                # 重置索引
                # Reset the index
                df_cleaned = df.drop(columns="bin").reset_index(drop=True)
                df_cleaned.attrs["farm_label"] = farm_label

                # 打印清洗结果
                # Print the cleaning results
                print(f"Remove Power Abandonment Data: {count_rule1}")
                print(f"Remove Measurement Error Data: {count_rule2}")

                # 返回清洗后的数据
                # Return the cleaned data
                return df_cleaned
