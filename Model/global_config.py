# 从 YAML 文件加载并统一提供全局参数
# Load and provide global parameters from the YAML file

from pathlib import Path

import yaml

# 根据当前文件位置确定全局参数文件路径
# Locate the global-parameter file from the current file
GLOBAL_PARAM_PATH = Path(__file__).resolve().with_name("global_param.yaml")

# 读取全局参数
# Read global parameters
with GLOBAL_PARAM_PATH.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# 检查全局参数文件内容
# Check the global-parameter file content
if not isinstance(config, dict) or not config:
    raise ValueError("global_param.yaml is empty or invalid.")
