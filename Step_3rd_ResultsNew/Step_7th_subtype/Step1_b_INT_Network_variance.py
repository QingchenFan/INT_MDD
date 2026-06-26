import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.conftest import matplotlib

'''
    计算两组亚型的 8 个网络水平的 variance，
    然后进行差异比较 (ANCOVA：控制 age, sex, mean_fd)，
    同时包含 3-Sigma 异常值剔除后的对比。
'''
matplotlib.use('Agg')

# ==========================================
# 1. 加载网络数据
# ==========================================
base_dir = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/'
df1 = pd.read_csv(base_dir + 'subtype1_INT_7net_agesex_FD.csv')
df2 = pd.read_csv(base_dir + 'subtype2_INT_7net_agesex_FD.csv')

# 提取用于计算方差的网络列（确保只包含这8个网络）
network_cols = ['subcortical', 'Visual', 'Somatomotor', 'Dorsal_Attention',
                'Ventral_Attention', 'Limbic', 'Frontoparietal', 'Default']

# ==========================================
# 2. 计算每个被试在 8 个网络上的方差，并合并数据
# ==========================================
df1['variance'] = df1[network_cols].var(axis=1)
df2['variance'] = df2[network_cols].var(axis=1)

df1['group'] = 'Subtype 1'
df2['group'] = 'Subtype 2'

# 提取后续统计分析需要的列
keep_cols = ['subID', 'group', 'age', 'sex', 'mean_fd', 'variance']
df_raw = pd.concat([df1[keep_cols], df2[keep_cols]], ignore_index=True).dropna()

# ==========================================
# 3. 识别离群点 (3-Sigma 准则) - 组内分别计算 Z-score
# ==========================================
# 按组别分别计算 variance 的 z-score
df_raw['variance_zscore'] = df_raw.groupby('group')['variance'].transform(lambda x: zscore(x, ddof=1))
# 过滤掉绝对值 > 3 的离群点
df_clean = df_raw[np.abs(df_raw['variance_zscore']) <= 3.0].copy()

print(f"原始数据样本量: {len(df_raw)}")
print(f"剔除 3-Sigma 离群点后的样本量: {len(df_clean)} (剔除了 {len(df_raw) - len(df_clean)} 个)")

# ==========================================
# 4. 统计检验：ANCOVA 控制协变量
# ==========================================
def run_ancova(data):
    """
    运行协方差分析，返回 ANOVA 表，以及组别差异的 F 值和 P 值
    """
    model = ols('variance ~ C(group) + age + C(sex) + mean_fd', data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    p_val = anova_table.loc['C(group)', 'PR(>F)']
    f_val = anova_table.loc['C(group)', 'F']
    return anova_table, f_val, p_val

anova_raw, f_raw, p_raw = run_ancova(df_raw)
anova_clean, f_clean, p_clean = run_ancova(df_clean)

print("\n========== ANCOVA 结果 (Original Data) ==========")
print(anova_raw)
print("\n========== ANCOVA 结果 (Cleaned Data) ==========")
print(anova_clean)

# ==========================================
# 5. 绘图对比
# ==========================================
plt.figure(figsize=(14, 6))

# --- 左图：原始数据分布 ---
plt.subplot(1, 2, 1)
sns.boxplot(x='group', y='variance', data=df_raw, palette='Set2', showfliers=True)
sns.stripplot(x='group', y='variance', data=df_raw, color='black', alpha=0.3, jitter=True)
# 将 ANCOVA 得到的真实 P 值打印在标题上
plt.title(f'Original Network Variance\nANCOVA (covFD, age, sex) p={p_raw:.4e}', fontsize=13)
plt.ylabel('Variance')
plt.xlabel('Subtype')

# --- 右图：剔除离群点后的分布 ---
plt.subplot(1, 2, 2)
sns.boxplot(x='group', y='variance', data=df_clean, palette='Pastel2', showfliers=False)
sns.stripplot(x='group', y='variance', data=df_clean, color='black', alpha=0.3, jitter=True)
plt.title(f'Cleaned Network Variance (>3SD Removed)\nANCOVA (covFD, age, sex) p={p_clean:.4e}', fontsize=13)
plt.ylabel('Variance')
plt.xlabel('Subtype')

plt.tight_layout()
save_path_img = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_INTVariance/Step1_VarianceDiff/network_variance_covFD_comparison.png'
plt.savefig(save_path_img, dpi=300)
plt.show()
print(f"\n绘图已保存至: {save_path_img}")

# 保存带有方差计算结果的文件
save_path_csv = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_INTVariance/Step1_VarianceDiff/subject_network_variances_covFD.csv'
df_raw.drop(columns=['variance_zscore']).to_csv(save_path_csv, index=False)
print(f"数据表已保存至: {save_path_csv}")