# 使用已保存的超参数训练各监测变量的 TFT 模型并保存检查点。
# Train TFT models for monitored variables with saved hyperparameters and store their checkpoints.


import json
import logging
from pathlib import Path
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, MAPE, RMSE, SMAPE, QuantileLoss

# 定义中转文件、模型权重和日志的相对路径
# Define relative paths for transit files, model weights, and logs
TRANSIT_DIR = Path(__file__).resolve().parents[2] / "Transit"
HYPERPARAM_PATH = TRANSIT_DIR / "model_hyperparams.json"
TFT_NET_DIR = TRANSIT_DIR / "tft_net"
TFT_TRAIN_LOG_DIR = TRANSIT_DIR / "logs" / "tft_train"

# 导入统一加载的全局参数
# Import centrally loaded global parameters
from global_config import config

# 禁用日志信息
# Disable log information
if config["disable_log_info"]:
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)


class CompletedBarHider(TQDMProgressBar):
    """Hide only the graphical bar after an epoch, while keeping its statistics."""

    _COMPLETED_FORMAT = (
        "{l_bar}{n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        # Restore tqdm's normal bar while the next epoch is running.
        if self._train_progress_bar is not None:
            self.train_progress_bar.bar_format = None

    def on_train_epoch_end(self, trainer, pl_module):
        # Removing {bar} changes only the completed line's visual format.
        # Counts, timing information and logged metrics remain visible.
        if self._train_progress_bar is not None:
            self.train_progress_bar.bar_format = self._COMPLETED_FORMAT
        super().on_train_epoch_end(trainer, pl_module)


class BuildTFTModel:
    @classmethod
    def train_tft(cls, datasets):

        torch.cuda.empty_cache()

        # 打印流程提示
        # Print process description
        print("\033[1;93m\n--- Process 7 ---  Training & Output Monitoring Variable (Offline)  ---\n\033[0m")

        # 读取模型超参数字典
        # Read tft hyperparam dictionary
        with HYPERPARAM_PATH.open("r", encoding="utf-8") as f:
            model_params = json.load(f)

        # 设置损失函数和输出维度
        # Setting the loss function and output dimensions
        quan_forecast = config["quan_forecast"]
        if quan_forecast["true_or_false"]:
            loss_function = QuantileLoss(quantiles=quan_forecast["quantile"])
            output_size = len(quan_forecast["quantile"])
        else:
            loss_function = RMSE()
            output_size = 1
        model_params.update({"loss": loss_function, "output_size": output_size})

        # 设置数据加载器参数
        # Set dataloader params
        batch_size = config["batch_size"]
        matmul_precision = config["matmul_precision"]
        num_workers = config["num_workers"]
        torch.set_float32_matmul_precision(matmul_precision)

        # 创建文件夹和字典来存储模型权重
        # Create a folder and a dictionary to store model weights
        TFT_NET_DIR.mkdir(parents=True, exist_ok=True)
        tfts = {}
        print("Loading a Shared Encoder Representation ...")
        print("\nIndependent Prediction Head for:")

        # 遍历每个监测变量的时间序列数据集
        # Traverse the time series dataset for each monitoring variable
        for monitoring_variable, dataset in datasets.items():

            # 打印提示
            # Printing tips
            print(f"\n    - {monitoring_variable} Starts Training ...")

            # 读取训练集和验证集
            # Read the training and validation
            training = dataset["training"]
            validation = dataset["validation"]

            # 设置训练器参数
            # Set trainer params
            trainer_params = {
                "max_epochs": config["max_epochs"],
                "devices": config["devices"],
                "accelerator": config["accelerator"],
                "precision": config["precision"],
                "num_sanity_val_steps": config["num_sanity_val_steps"],
                "log_every_n_steps": config["log_every_n_steps"],
                "logger": config["logger"],
                "enable_progress_bar": config["enable_progress_bar"],
                "enable_model_summary": config["enable_model_summary"],
                "enable_checkpointing": False,
                "default_root_dir": str(TFT_TRAIN_LOG_DIR),
            }
            if config["enable_progress_bar"]:
                trainer_params["callbacks"] = [CompletedBarHider()]

            # 为训练集和验证集创造数据加载器
            # Creating dataLoaders for train set and validation set
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
            tfts[monitoring_variable] = TemporalFusionTransformer.from_dataset(
                training,
                **model_params,
                logging_metrics=[SMAPE(), MAE(), RMSE(), MAPE()],
            )
            trainer = Trainer(**trainer_params)
            trainer.fit(
                tfts[monitoring_variable],
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )

        # 保存训练好的模型
        # Save the trained model
            model_path = TFT_NET_DIR / f"{monitoring_variable}.ckpt"
            trainer.save_checkpoint(str(model_path))
        print("\nTraining is Finished ...")
        print(f"TFT Model Saved to {TFT_NET_DIR}!")
