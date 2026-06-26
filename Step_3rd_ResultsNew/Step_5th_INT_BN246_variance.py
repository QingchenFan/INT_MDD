import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.conftest import matplotlib

matplotlib.use('Agg')

# 1. 加载数据
df1 = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype1_INT.csv')
df2 = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype2_INT.csv')

# 提取脑区列（排除 subID）
region_cols = [col for col in df1.columns if col != 'subID']

# 2. 计算每个被试的方差
var1 = df1[region_cols].var(axis=1)
var2 = df2[region_cols].var(axis=1)

# 3. 识别并处理离群点 (以 Subtype 1 为主)
mean1, std1 = var1.mean(), var1.std()
# 寻找超出 3 倍标准差的索引
is_outlier = abs(var1 - mean1) > 3 * std1
outliers = var1[is_outlier]
var1_cleaned = var1[~is_outlier]

print(f"--- 离群点分析 ---")
print(f"Subtype 1 原始样本量: {len(var1)}")
print(f"识别到的离群点个数: {len(outliers)}")
if len(outliers) > 0:
    print(f"离群点被试 ID: {df1.loc[outliers.index, 'subID'].values}")
    print(f"离群点方差值: {outliers.values}")

# 4. 统计检验对比
print(f"\n--- 统计检验结果 ---")

# (A) 包含离群点的原始 T 检验
t_orig, p_orig = stats.ttest_ind(var1, var2)
print(f"1. 原始双样本 T 检验: t = {t_orig:.4f}, p = {p_orig:.4f}")

# (B) 剔除离群点后的 T 检验
t_clean, p_clean = stats.ttest_ind(var1_cleaned, var2)
print(f"2. 剔除离群点后 T 检验: t = {t_clean:.4f}, p = {p_clean:.4f}")

# (C) 非参数检验 (Mann-Whitney U，对离群点不敏感，建议作为最终依据)
u_stat, p_u = stats.mannwhitneyu(var1, var2, alternative='two-sided')
print(f"3. 曼-惠特尼 U 检验 (非参数): U = {u_stat:.4f}, p = {p_u:.4f}")

# 5. 可视化：清洗前后的对比图
plt.figure(figsize=(14, 6))

# 图 1：包含离群点
plt.subplot(1, 2, 1)
data_orig = pd.DataFrame({
    'Variance': pd.concat([var1, var2]),
    'Group': ['Subtype 1 (Orig)']*len(var1) + ['Subtype 2']*len(var2)
})
sns.boxplot(x='Group', y='Variance', data=data_orig, palette='Set2', showfliers=True)
sns.stripplot(x='Group', y='Variance', data=data_orig, color='black', alpha=0.3, jitter=True)
plt.title(f'Original Data\nT-test p={p_orig:.4f}')

# 图 2：剔除离群点后
plt.subplot(1, 2, 2)
data_clean = pd.DataFrame({
    'Variance': pd.concat([var1_cleaned, var2]),
    'Group': ['Subtype 1 (Cleaned)']*len(var1_cleaned) + ['Subtype 2']*len(var2)
})
sns.boxplot(x='Group', y='Variance', data=data_clean, palette='Pastel2', showfliers=False)
sns.stripplot(x='Group', y='Variance', data=data_clean, color='black', alpha=0.3, jitter=True)
plt.title(f'Cleaned Data (3-Sigma Outliers Removed)\nT-test p={p_clean:.4f}')

plt.tight_layout()
plt.savefig('variance_robustness_check.png')
plt.show()

# 6. 导出清洗后的数据表（可选）
# result_cleaned = pd.concat([
#     pd.DataFrame({'subID': df1.loc[var1_cleaned.index, 'subID'], 'variance': var1_cleaned, 'subtype': 'Subtype 1'}),
#     pd.DataFrame({'subID': df2['subID'], 'variance': var2, 'subtype': 'Subtype 2'})
# ])
# result_cleaned.to_csv('cleaned_subject_variances.csv', index=False)