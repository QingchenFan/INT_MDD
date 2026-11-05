import pandas as pd
import numpy as np
from scipy import stats


# 定义计算 Cohen's d 的函数
def calculate_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # 计算合并方差 (pooled variance)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

    # 计算 Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / np.sqrt(pooled_var)
    return d


# 1. 加载数据
# hc_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step2_Ttest/HC_INT20_7net_agesex.csv')
# mdd_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step2_Ttest/MDD_INT20_7net_agesex.csv')

hc_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_Match/Step2_subtype/subtype1_INT20_7net_agesex.csv')
mdd_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_Match/Step2_subtype/subtype2_INT20_7net_agesex.csv')

# --- 2. 年龄 (Age) 的 t 检验 ---
# 提取年龄数据并排除缺失值
hc_age = hc_df['age'].dropna()
mdd_age = mdd_df['age'].dropna()

# 执行独立样本 t 检验 (使用 Welch's t-test，不假设方差齐性)
t_stat, p_val_age = stats.ttest_ind(hc_age, mdd_age, equal_var=False)

# 计算年龄的 Cohen's d 效应量
cohens_d_age = calculate_cohens_d(hc_age, mdd_age)

# --- 3. 性别 (Sex) 的卡方检验 ---
# 统计两组中各性别的频数
hc_sex_counts = hc_df['sex'].value_counts().sort_index()
mdd_sex_counts = mdd_df['sex'].value_counts().sort_index()

# 构建列联表 (Contingency Table)
contingency_table = pd.DataFrame([hc_sex_counts, mdd_sex_counts], index=['HC', 'MDD']).fillna(0)

# 执行卡方检验
chi2, p_val_sex, dof, expected = stats.chi2_contingency(contingency_table)

# 针对卡方检验，计算 Cramer's V 效应量 (可选补充)
n_total = contingency_table.sum().sum()
min_dim = min(contingency_table.shape) - 1
cramers_v = np.sqrt(chi2 / (n_total * min_dim)) if min_dim > 0 else 0

# --- 4. 打印结果 ---
print("=== 年龄 (Age) t 检验结果 ===")
print(f"健康对照组 (HC): {hc_age.mean():.2f} ± {hc_age.std():.2f}")
print(f"重度抑郁组 (MDD): {mdd_age.mean():.2f} ± {mdd_age.std():.2f}")
print(f"t 统计量: {t_stat:.4f}, p 值: {p_val_age:.4f}")
print(f"Cohen's d 效应量: {cohens_d_age:.4f} (解释: 0.2=小, 0.5=中, 0.8=大)")

print("\n=== 性别 (Sex) 卡方检验结果 ===")
print("性别频数统计表 (1=男, 2=女/或其他码值):")
print(contingency_table)
print(f"卡方统计量: {chi2:.4f}, p 值: {p_val_sex:.4f}")
print(f"Cramer's V 效应量: {cramers_v:.4f}")