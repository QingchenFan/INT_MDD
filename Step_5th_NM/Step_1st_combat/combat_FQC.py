import pandas as pd
import numpy as np
from neuroHarmonize import harmonizationLearn, harmonizationApply

# ---------------------- 1. 读取数据（请根据实际路径修改）----------------------
HC = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC_MDD/INT_NM/Feature/INT_HC.csv')
MDD = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC_MDD/INT_NM/Feature/INT_all.csv')


# ---------------------- 2. 数据预处理：分离特征与协变量 ----------------------
# 定义协变量列（SITE、age、sex）
covar_cols = ['SITE', 'age', 'sex']

# 定义标识列（保留subID、age、sex、SITE，同时为MDD增加disorder）
id_cols = ['subID', 'age', 'sex', 'SITE']  # 基础标识列
mdd_id_cols = id_cols + ['disorder']  # MDD额外保留disorder列

# 定义特征列：排除标识列，仅保留数值型特征
feature_cols = [col for col in HC.columns if col not in id_cols]


# ---------------------- 3. 处理所有数据集：转换为numpy数组 ----------------------
# HC数据集
HC_features = HC[feature_cols].values
HC_covars = HC[covar_cols]

# MDD数据集
MDD_features = MDD[feature_cols].values
MDD_covars = MDD[covar_cols]


print("\n🚀 开始用所有HC样本训练ComBat模型...")
model, data_norm_harmonized = harmonizationLearn(
    data=HC_features,
    covars=HC_covars
)


data_MDD_harmonized = harmonizationApply(
    data=MDD_features,
    covars=MDD_covars,
    model=model
)


# ---------------------- 4. 保存 harmonized 结果 ----------------------
# 处理HC的harmonized结果
hc_harmonized_df = pd.DataFrame(
    data=data_norm_harmonized,
    columns=feature_cols
)
hc_harmonized_df = pd.concat([HC[id_cols].reset_index(drop=True), hc_harmonized_df], axis=1)
hc_harmonized_df.to_csv('./HC_harmonized.csv', index=False)
print("✅ HC harmonized数据已保存为 HC_harmonized.csv")


# 处理MDD的harmonized结果（增加disorder列）
mdd_harmonized_df = pd.DataFrame(
    data=data_MDD_harmonized,
    columns=feature_cols
)
# 合并标识信息（包含disorder）
mdd_harmonized_df = pd.concat([MDD[mdd_id_cols].reset_index(drop=True), mdd_harmonized_df], axis=1)
mdd_harmonized_df.to_csv('./MDD_harmonized.csv', index=False)
print("✅ MDD harmonized数据已保存为 MDD_harmonized.csv（包含disorder列）")