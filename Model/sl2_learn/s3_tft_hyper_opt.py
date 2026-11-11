# 使用固定配置或 Optuna 搜索 TFT 超参数，并保存参数字典。
# Use fixed settings or Optuna to optimize TFT hyperparameters and save the parameter dictionary.


import json
import logging
from pathlib import Path
import optuna
import torch
from lightning.pytorch import Trainer
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE

# 定义中转文件和日志的相对路径
# Define relative paths for transit files and logs
TRANSIT_DIR = Path(__file__).resolve().parents[2] / "Transit"
HYPERPARAM_PATH = TRANSIT_DIR / "model_hyperparams.json"
HYPEROPT_LOG_DIR = TRANSIT_DIR / "logs" / "hyperopt"

# 导入统一加载的全局参数
# Import centrally loaded global parameters
from global_config import config

# 禁用日志信息
# Disable log information
if config["disable_log_info"]:
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)


# 构造TFT训练函数
# Build TFT train function
def train(
    datasets,
    trial: optuna.Trial,
    lstm_layers,
    hidden_size,
    attention_head_size,
    dropout,
    hidden_continuous_size,
) -> float:

    # 获取首要监测变量对应的时间序列数据集
    # Obtain time series datasets of key monitoring variables
    dataset = next(iter(datasets.values()))
    training = dataset["training"]
    validation = dataset["validation"]

    # 设置TFT模型的架构参数
    # Set the architecture params of the tft model
    model_params = {
        # 待优化超参数
        # Hyperparams to be optimized
        "hidden_size": hidden_size,
        "lstm_layers": lstm_layers,
        "attention_head_size": attention_head_size,
        "dropout": dropout,
        "hidden_continuous_size": hidden_continuous_size,
        # 直接继承的超参数
        # Directly inherited hyperparams
        "log_interval": config["log_interval"],
        "reduce_on_plateau_patience": config["reduce_on_plateau_patience"],
        "optimizer": config["optimizer"],
        "learning_rate": config["learning_rate"],
        # 输出维度为1以提高效率
        # The output dimension is 1 for better efficiency
        "output_size": 1,
        # 使用均方误差作为损失函数以配合输出维度
        # Use RMSE as the loss function to match the output dimensions
        "loss": RMSE(),
        # 回调模型效果指标
        # Call back the model's indicators
        "logging_metrics": [RMSE()],
    }

    # 设置训练器的参数
    # Set the trainer params
    trainer_params = {
        "max_epochs": config["max_epochs_of_hyperopt"],
        "devices": 1,
        "accelerator": config["accelerator"],
        "precision": config["precision"],
        "num_sanity_val_steps": 0,
        "log_every_n_steps": -1,
        "logger": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "enable_checkpointing": False,
        "default_root_dir": str(HYPEROPT_LOG_DIR),
    }

    # 设置数据加载器的参数
    # Set the dataloader params
    batch_size = config["batch_size"]
    matmul_precision = config["matmul_precision"]
    num_workers = config["num_workers"]
    torch.set_float32_matmul_precision(matmul_precision)

    # 为训练集和验证集创建数据加载器
    # Create dataLoaders for train set and validation set
    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # 构建模型并训练
    # Build the model and train it
    tft = TemporalFusionTransformer.from_dataset(training, **model_params)
    trainer = Trainer(**trainer_params)
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # 回调最终的验证损失
    # Callback of the final validation loss
    rmse = trainer.callback_metrics.get("val_loss")

    # 将验证损失转换为 Optuna 要求的 Python 浮点数
    # Convert the validation loss to the Python float required by Optuna
    if rmse is None:
        return float("inf")
    if isinstance(rmse, torch.Tensor):
        return float(rmse.detach().cpu().item())
    return float(rmse)


# 构造解空间搜索函数
# Construct solution space search function
def build_objective(datasets):
    def objective(trial: optuna.Trial) -> float:

        # 设置解空间
        # Set search space
        lstm_layers = trial.suggest_int("lstm_layers", 1, 4)
        hidden_size = trial.suggest_int("hidden_size", 16, 256, step=16)
        attention_head_size = trial.suggest_int("attention_head_size", 2, 8)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        hidden_continuous_size = trial.suggest_int("hidden_continuous_size", 8, 128, step=8)

        # 设置目标函数
        # Set the objective function
        rmse = train(
            datasets,
            trial,
            lstm_layers,
            hidden_size,
            attention_head_size,
            dropout,
            hidden_continuous_size,
        )

        # 打印当前试验的搜索结果
        # Print the results of the current trial
        print(
            f"    - Trial {trial.number}:  Params are {lstm_layers}, "
            f"{hidden_size}, {dropout * 100:.1f}%, etc.; With RMSE of {rmse}"
        )

        # 返回闭包结果
        # Return closure result
        return rmse

    return objective


class HyperparamOpt:
    @classmethod
    def hyper_opt(cls, datasets):

        # 打印流程提示
        # Print process description
        print("\033[1;93m\n--- Process 6 ---  Optimizing TFT Hyperparams  ---\033[0m")

        # 根据自动调参开关的布尔值判断：使用全局超参数
        # Judging by the Boolean value of automatic_conf: Use global hyperparams
        if not config["automatic_conf"]:
            model_params = {
                "hidden_size": config["hidden_size"],
                "lstm_layers": config["lstm_layers"],
                "attention_head_size": config["attention_head_size"],
                "dropout": config["dropout"],
                "hidden_continuous_size": config["hidden_continuous_size"],
                "log_interval": config["log_interval"],
                "reduce_on_plateau_patience": config["reduce_on_plateau_patience"],
                "optimizer": config["optimizer"],
                "learning_rate": config["learning_rate"],
            }

            # 打印提示
            # Printing tips
            print("\nUse Fixed Configuration Hyperparams!")

            # 定义文件路径
            # Define the file path
            TRANSIT_DIR.mkdir(parents=True, exist_ok=True)
            file_path = HYPERPARAM_PATH

            # 将超参数字典保存为［JSON］文件
            # Save the hyperparam dictionary as a JSON file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(model_params, f, ensure_ascii=False, indent=4)
            print("\nHyperparam Dictionary Saved!")

            # 返回全局超参数字典
            # Return the global hyperparam dictionary
            return model_params

        # 根据自动调参开关的布尔值判断：使用自动寻优的超参数
        # Judging by the Boolean value of automatic_conf: Use automatically opt hyperparams
        else:

            # 打印提示
            # Printing tips
            print("\nStart optimizing hyperparams ...")

            # Reuse a non-empty hyperparameter file beside this script.
            file_path = HYPERPARAM_PATH
            if file_path.is_file() and file_path.stat().st_size > 0:
                try:
                    with file_path.open("r", encoding="utf-8") as f:
                        model_params = json.load(f)
                    required_params = {
                        "lstm_layers",
                        "hidden_size",
                        "attention_head_size",
                        "dropout",
                        "hidden_continuous_size",
                    }
                    if not required_params.issubset(model_params):
                        raise ValueError("The saved hyperparameter file is incomplete.")

                    lstm_layers = model_params["lstm_layers"]
                    hidden_size = model_params["hidden_size"]
                    attention_head_size = model_params["attention_head_size"]
                    dropout = model_params["dropout"]
                    hidden_continuous_size = model_params["hidden_continuous_size"]

                    print("\nOptimization Result:")
                    print(f"LSTM_Layers: {lstm_layers}")
                    print(f"Hidden Size: {hidden_size}")
                    print(f"Attent Size: {attention_head_size}")
                    print(f"Dropout Rat: {dropout}")
                    print(f"Hidden Cont: {hidden_continuous_size}")
                    print("\nHyperparams Optimization is Finished ...")
                    return model_params
                except (json.JSONDecodeError, OSError, TypeError, ValueError):
                    # Invalid or incomplete content cannot be reused.
                    pass

            # 调用解空间搜索函数
            # Call solution space search function
            study = optuna.create_study(direction="minimize")
            objective_fn = build_objective(datasets)
            study.optimize(objective_fn, n_trials=config["trials_epoch"])

            # 超参数求解结果
            # Hyperparam solution results
            best_params = study.best_params
            lstm_layers = best_params["lstm_layers"]
            hidden_size = best_params["hidden_size"]
            attention_head_size = best_params["attention_head_size"]
            dropout = best_params["dropout"]
            hidden_continuous_size = best_params["hidden_continuous_size"]

            # 将求解结果加入模型超参数字典
            # Add the solution results to the tft hyperparam dictionary
            model_params = {
                "hidden_size": hidden_size,
                "lstm_layers": lstm_layers,
                "attention_head_size": attention_head_size,
                "dropout": dropout,
                "hidden_continuous_size": hidden_continuous_size,
                "log_interval": config["log_interval"],
                "reduce_on_plateau_patience": config["reduce_on_plateau_patience"],
                "optimizer": config["optimizer"],
                "learning_rate": config["learning_rate"],
            }

            # 打印超参数寻优结果
            # Print hyperparam opt result
            print("\nOptimization Result:")
            print(f"LSTM_Layers: {lstm_layers}")
            print(f"Hidden Size: {hidden_size}")
            print(f"Attent Size: {attention_head_size}")
            print(f"Dropout Rat: {dropout}")
            print(f"Hidden Cont: {hidden_continuous_size}")
            print("\nHyperparams Optimization is Finished ...")

            # 定义文件路径
            # Define the file path
            TRANSIT_DIR.mkdir(parents=True, exist_ok=True)
            file_path = HYPERPARAM_PATH

            # 将超参数字典保存为［JSON］文件
            # Save the hyperparam dictionary as a JSON file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(model_params, f, ensure_ascii=False, indent=4)
            print("Hyperparam Dictionary Saved!")

            # 返回寻优后的超参数字典
            # Return the opt hyperparam dictionary
            return model_params
