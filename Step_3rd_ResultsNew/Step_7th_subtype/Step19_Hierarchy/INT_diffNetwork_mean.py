import pandas as pd
import numpy as np
import os

# ----------------- 1. 文件路径设置 -----------------
# 请根据你的实际路径修改
input_file = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_NetworkHierarchy/INT_DiffNetwork/HC_DiffNetwork_flattened.csv'
output_file = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_NetworkHierarchy/INT_DiffNetwork/HC_DiffNetwork_mean.csv'

# ----------------- 2. 读取数据 -----------------
df = pd.read_csv(input_file)

# ----------------- 3. 列名划分 -----------------
# 明确需要保留的协变量列名
covariate_cols = ['subID', 'age', 'sex', 'mean_fd']

# 动态提取所有网络边的列名（即除了协变量之外的所有列）
edge_cols = [col for col in df.columns if col not in covariate_cols]

# ----------------- 4. 计算绝对值后的均值 -----------------
# 仅提取 28 个网络差值的数据
edges_data = df[edge_cols]

# 对差值取绝对值，然后按行（axis=1，即计算每个被试的均值）求平均
# 这个新指标代表了每个被试的 "全局网络分化度 (Global Network Differentiation)"
global_diff_mean = edges_data.abs().mean(axis=1)

# ----------------- 5. 结果拼接与保存 -----------------
# 提取原始的协变量数据
df_result = df[covariate_cols].copy()

# 将计算好的均值作为新的一列加入
df_result['Global_Diff_Mean'] = global_diff_mean

# 保存到新的 CSV 文件中
df_result.to_csv(output_file, index=False)

