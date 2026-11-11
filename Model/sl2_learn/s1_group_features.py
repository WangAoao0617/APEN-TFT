# 根据传感变量间的 Pearson 相关性构建分组结果。
# Group sensor variables according to their pairwise Pearson correlations.

from collections import defaultdict
from itertools import combinations
import pandas as pd
from global_config import config


class GroupFeature:
    @classmethod
    def group_var(cls, df):

        # 调用全局参数
        # Calling global params
        threshold = config["pccs_threshold"]
        farm_label = config["wind_farm_id"]

        # 打印流程提示
        # Print process description
        print("\033[1;93m\n--- Process 4 ---  Clustering Colline Sensors  ---\033[0m")

        # 从对应风场的特征说明表［comma_feature_description.csv］提取传感参数名
        # Extract the sensor name from the farm_label's comma_feature_description.csv
        info = pd.read_csv(
            f"../Archive/Wind Farm {farm_label}/comma_feature_description.csv"
        )
        variable_list = info["sensor_name"].tolist()

        # 创建变量到数据框列名的映射
        # Create a mapping of variables to df column names
        var_to_col = {}
        for var in variable_list:
            matching_cols = [
                col for col in df.columns if col == var or col == f"{var}_avg"
            ]
            if len(matching_cols) == 1:
                var_to_col[var] = matching_cols[0]
            else:
                raise ValueError(
                    "Matching error, please check the uniqueness and existence of the variable!"
                )

        # 提取相关列
        # Extract relevant columns
        selected_cols = [var_to_col[var] for var in variable_list]
        subset_df = df[selected_cols]

        # 调用皮尔逊相关系数计算方法
        # Call the PCCs calculation method
        corr_matrix = subset_df.corr(method="pearson").abs()

        # 两两计算皮尔逊相关系数并构建邻接矩阵
        # Calculate the PCCs pairwise and construct the adjacency matrix
        adjacency = {var: set() for var in variable_list}
        for var1, var2 in combinations(variable_list, 2):
            corr_value = corr_matrix.loc[var_to_col[var1], var_to_col[var2]]
            if corr_value >= threshold:
                adjacency[var1].add(var2)
                adjacency[var2].add(var1)

        # 使用深度优先搜索算法查找所有连通分量
        # Use DFS to find all connected components
        groups = {}
        visited = set()
        current_group = 0

        # 深度优先搜索
        # Depth First Search
        for var in variable_list:
            if var not in visited:
                stack = [var]
                groups[var] = current_group
                visited.add(var)
                while stack:
                    current_var = stack.pop()
                    for neighbor in adjacency[current_var]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            groups[neighbor] = current_group
                            stack.append(neighbor)
                current_group += 1

        # 构建组标签到变量的映射
        # Constructing a mapping of group labels to variables
        group_to_vars = defaultdict(list)
        for var, group in groups.items():
            group_to_vars[group].append(var)

        # 过滤出有多个变量的组
        # Filter out groups with multiple variables
        multi_var_groups = {
            group: vars_in_group
            for group, vars_in_group in group_to_vars.items()
            if len(vars_in_group) > 1
        }

        # 对于多变量组，找到这些组的最大变量数
        # For multivariate groups, find the maximum number of variables in each group
        if multi_var_groups:
            max_vars = max(
                len(vars_in_group) for vars_in_group in multi_var_groups.values()
            )

            # 构建每一行数据
            # Constructing each row of data
            group_rows = []
            for group, vars_in_group in sorted(multi_var_groups.items()):
                row = [group] + vars_in_group

                # 填补空白以匹配最大数量的变量
                # Fill the gaps to match the maximum number of variables
                row += ["--"] * (max_vars - len(vars_in_group))
                group_rows.append(row)

            # 创建列名
            # Create column names
            column_names = ["Group"] + [f"Variable_{i + 1}" for i in range(max_vars)]

            # 创建数据框
            # Creating a DataFrame
            groups_df = pd.DataFrame(group_rows, columns=column_names)

            # 创建最多包含前五列的分组信息预览
            # Create a grouped-feature preview containing at most the first five columns
            groups_preview_df = groups_df.iloc[:, :5].copy()
            if groups_df.shape[1] > 5:
                groups_preview_df["..."] = "..."

            # 打印分组信息预览和原始数据框尺寸
            # Print the grouped-feature preview and the original DataFrame dimensions
            print("\nClustered Features:")
            print(groups_preview_df.to_string(index=False))
            print(f"[{groups_df.shape[0]} rows x {groups_df.shape[1]} columns]")

        # 构建分组结果列表
        # Build grouping result list
        var_group_list = [[var, groups[var]] for var in variable_list]

        # 返回分组结果列表
        # Return grouping result list
        return var_group_list
