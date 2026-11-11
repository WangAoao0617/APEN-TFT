# 这里是主程序。
# This is the main programme.

import os
from pathlib import Path
import warnings
import pandas as pd

if __name__ == "__main__":

    # 将工作目录固定为入口脚本所在目录
    # Set the working directory to the directory containing the entry script
    os.chdir(Path(__file__).resolve().parent)

    # 隐藏由第三方 TFT 模型重复记录损失模块引起的非关键警告
    # Hide the noncritical warning caused by the third-party TFT model recording its loss module twice
    warnings.filterwarnings(
        "ignore",
        message=r"Attribute 'loss' is an instance of `nn\.Module` and is already saved during checkpointing\.",
        category=UserWarning,
        module=r"lightning\.pytorch\.utilities\.parsing",
    )

    from sl1_load import LoadData, CleanData, FormDataset
    from sl2_learn import GroupFeature, CreateTimeSeriesSet, HyperparamOpt, BuildTFTModel, TFTOnlineLearning
    from sl3_alarm import TFTFaultWarning

    df = LoadData.load_data()
    df = CleanData.cln_data(df)
    df_list = FormDataset.form_dataset(df)
    var_list = GroupFeature.group_var(df)
    tsset = CreateTimeSeriesSet.create_datasets(df_list, var_list)
    params = HyperparamOpt.hyper_opt(tsset)
    model = BuildTFTModel.train_tft(tsset)
    
    online_df = pd.concat(df_list, ignore_index=True)
    online_df = online_df.sort_values("time_idx").reset_index(drop=True)
    online = TFTOnlineLearning.online_learning(online_df, var_list)

    alarm_results = TFTFaultWarning.predict_and_warn(tsset, df_list[1])
