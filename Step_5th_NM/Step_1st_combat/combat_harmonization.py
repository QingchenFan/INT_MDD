import os
import pandas as pd
import numpy as np
from neuroHarmonize import harmonizationLearn, harmonizationApply

# =============================
# 🚀 0. 路径配置
# =============================
INPUT_CSV = '/Volumes/QC/INT/INT_BN246_HC_MDD/INT_NM/Feature/combat_test.csv'
OUT_DIR = './combat_results_harmonize_final'
os.makedirs(OUT_DIR, exist_ok=True)

# =============================
# 🚀 1. 读取数据
# =============================
alldata = pd.read_csv(INPUT_CSV)
print("✅ 数据读取完成")
print("总样本数:", len(alldata))
print("列名:", alldata.columns.tolist())

# =============================
# 🚀 2. 基础列名定义
# =============================
site_col = 'sitename'        # 站点名称（如 AD135, DZ, HCP）
batch_col = 'SITE'    # 批次变量（1~4）
group_col = 'disorder'   # 分组变量（1=HC，2=MDD）
age_col, sex_col = 'age', 'sex'  # 协变量

# =============================
# 🚀 3. 特征列识别
# =============================
feature_cols = alldata.columns[5:]  # 前4列为元信息，其余为特征
data_all = alldata[feature_cols].to_numpy()  # neuroHarmonize 要求 (n_samples × n_features)

# =============================
# 🚀 4. 定义样本集合
# =============================
is_hc = alldata[group_col] == 1
is_mdd = alldata[group_col] == 2
is_ad135 = alldata[site_col] == 'AD135'
is_dz = alldata[site_col] == 'DZ'
is_hcp = alldata[site_col] == 'HCP'



norm_mask = is_hc  # 常模构建集：所有站点的HC
discovery_mask = is_mdd & is_ad135  # 发现集：AD135的MDD
test_mask = is_mdd & is_dz  # 验证集：DZ的MDD

remaining_mdd_mask = is_mdd & (~(discovery_mask | test_mask))
print(remaining_mdd_mask)
# =============================
# 🚀 5. 样本量检查
# =============================
print("\n📊 样本量统计:")
print(f"常模构建集（所有站点HC）: {norm_mask.sum()}")
print(f"发现集（AD135 MDD）: {discovery_mask.sum()}")
print(f"验证集（DZ MDD）: {test_mask.sum()}")
print(f"剩余 MDD（AD HX MDD）: {remaining_mdd_mask.sum()}")

# # 检查无样本重叠
# assert (norm_mask & discovery_mask).sum() == 0, "❌ 常模集与发现集重叠！"
# assert (norm_mask & test_mask).sum() == 0, "❌ 常模集与验证集重叠！"
# assert (discovery_mask & test_mask).sum() == 0, "❌ 发现集与验证集重叠！"

# 输出各站点HC数量
print("\n各站点HC样本数量:")
print(alldata.loc[is_hc].groupby(site_col).size())

# =============================
# 🚀 6. 提取数据与协变量（核心修复：列名改为 SITE，匹配 neuroHarmonize 要求）
# =============================
def get_data_covars(mask):
    data = data_all[mask, :]
    # 关键修改：将批次列重命名为 SITE（工具强制要求）
    covars = alldata.loc[mask, [batch_col, age_col, sex_col]]
    return data, covars.reset_index(drop=True)

data_norm, covars_norm = get_data_covars(norm_mask)

data_discovery, covars_discovery = get_data_covars(discovery_mask)
data_test, covars_test = get_data_covars(test_mask)
data_remaining, covars_remaining = get_data_covars(remaining_mdd_mask)

# 验证协变量列名（确保包含 SITE）
print("\n协变量列名检查:")
print(f"常模集协变量列: {covars_norm.columns.tolist()}")  # 应包含 ['SITE', 'age', 'sex']

# =============================
# 🚀 7. 拟合 ComBat 模型（仅HC样本）
# =============================

print("\n🚀 开始用所有HC样本训练ComBat模型...")
model, data_norm_harmonized = harmonizationLearn(
    data=data_norm,
    covars=covars_norm  # 此时 covars 包含 SITE 列，符合要求
)
print("✅ ComBat模型训练完成！")

# =============================
# 🚀 8. 应用模型到发现集和验证集（MDD）
# =============================
print("\n🚀 校正发现集 MDD 样本（AD135）...")
data_discovery_harmonized = harmonizationApply(
    data=data_discovery,
    covars=covars_discovery,
    model=model
)

print("🚀 校正验证集 MDD 样本（DZ）...")
data_test_harmonized = harmonizationApply(
    data=data_test,
    covars=covars_test,
    model=model
)
print("\n🚀 校正剩余 MDD 样本（HX, AD 等站点）...")
data_remaining_harmonized = harmonizationApply(
    data=data_remaining,
    covars=covars_remaining,
    model=model
)
print("✅ 剩余 MDD 样本校正完成！")

print("✅ 所有样本校正完成！")

# =============================
# 🚀 9. 合并元数据与校正特征
# =============================
def merge_output(mask, data_harmonized):
    meta = alldata.loc[mask, [site_col, batch_col, group_col, age_col, sex_col]].reset_index(drop=True)
    df_data = pd.DataFrame(data_harmonized, columns=feature_cols)
    return pd.concat([meta, df_data], axis=1)

out_norm = merge_output(norm_mask, data_norm_harmonized)
out_discovery = merge_output(discovery_mask, data_discovery_harmonized)
out_test = merge_output(test_mask, data_test_harmonized)
out_remaining = merge_output(remaining_mdd_mask, data_remaining_harmonized)


print("✅ 剩余 MDD 校正结果已保存:", os.path.join(OUT_DIR, 'remaining_mdd_harmonized.csv'))

# =============================
# 🚀 10. 保存结果
# =============================
out_norm.to_csv(os.path.join(OUT_DIR, 'norm_harmonized.csv'), index=False)
out_discovery.to_csv(os.path.join(OUT_DIR, 'discovery_harmonized.csv'), index=False)
out_test.to_csv(os.path.join(OUT_DIR, 'test_harmonized.csv'), index=False)
out_remaining.to_csv(os.path.join(OUT_DIR, 'remaining_mdd_harmonized.csv'), index=False)
print("\n✅ 结果已保存：")
print(f"- 常模构建集（所有HC）: {os.path.join(OUT_DIR, 'norm_harmonized.csv')}")
print(f"- 发现集（AD135 MDD）: {os.path.join(OUT_DIR, 'discovery_harmonized.csv')}")
print(f"- 验证集（DZ MDD）: {os.path.join(OUT_DIR, 'test_harmonized.csv')}")
print("\n🎯 neuroHarmonize 站点效应校正流程全部完成！")