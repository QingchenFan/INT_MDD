import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.anova import AnovaRM  # 【新增】用于重复测量方差分析
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

'''
    计算被试内部 8 个网络间的 INT 差异，执行 RM-ANOVA、配对T检验及 FDR 校正，并画出热力图与箱线图
'''
# 解决非交互环境下的绘图问题
plt.rcParams['figure.max_open_warning'] = 50
import matplotlib
matplotlib.use('Agg')

# 1. 加载数据
df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_age/INT_40/Step2_subtype/subtype2_INT_7net_agesex_FD.csv')

# 2. 数据转换
# 【核心修改 1】：直接明确指定 8 个网络，完美避开 age, sex, mean_fd
networks = [
    'subcortical', 'Visual', 'Somatomotor', 'Dorsal_Attention',
    'Ventral_Attention', 'Limbic', 'Frontoparietal', 'Default'
]

# 宽表转长表，保留 subID 供被试内分析使用
df_melt = df.melt(id_vars=['subID'], value_vars=networks, var_name='Network', value_name='Value')

# 3. 重复测量方差分析 (RM-ANOVA)
# 【核心修改 2】：采用被试内设计的方差分析，代替原本的独立样本 ANOVA
rm_anova = AnovaRM(data=df_melt, depvar='Value', subject='subID', within=['Network']).fit()
anova_table = rm_anova.anova_table
# 稍微改一下保存的文件名以作区分
anova_table.to_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_age/INT_40/Step2_subtype/Step4_NetworkHierarchy/subtype2/subtype2_rm_anova_results.csv')

print("RM-ANOVA 完成，结果已保存为 HC_rm_anova_results.csv")
print("\nRM-ANOVA 统计表:")
print(anova_table)
print("-" * 50)

# 4. 事后检验 (配对 T 检验) + FDR 校正
pairs = list(combinations(networks, 2))
results_list = []

for g1, g2 in pairs:
    data1 = df[g1]
    data2 = df[g2]
    # stats.ttest_rel 是正确的配对 T 检验，无需修改
    stat, p_raw = stats.ttest_rel(data1, data2)
    results_list.append({
        'group1': g1,
        'group2': g2,
        'meandiff': data2.mean() - data1.mean(),
        't_stat': stat,
        'p_raw': p_raw
    })

# FDR 校正
p_values = [r['p_raw'] for r in results_list]
reject, p_fdr, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

posthoc_df = pd.DataFrame(results_list)
posthoc_df['p_fdr'] = p_fdr
posthoc_df['reject_fdr'] = reject
posthoc_df = posthoc_df.sort_values('p_fdr').reset_index(drop=True)
posthoc_df.to_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_age/INT_40/Step2_subtype/Step4_NetworkHierarchy/subtype2/subtype2_posthoc_fdr_results.csv', index=False)

print("事后检验 + FDR 完成，结果已保存为 posthoc_fdr_results.csv")
print(f"共有 {posthoc_df['reject_fdr'].sum()} 对在FDR校正后显著 (p<0.05)\n")


# ====================== 5. 简单箱体图（无显著性标记） ======================
order = networks

plt.figure(figsize=(14, 8))
sns.set(style="whitegrid")

ax = sns.boxplot(x='Network', y='Value', data=df_melt,
                 order=order, palette='Set3', width=0.65)

sns.stripplot(x='Network', y='Value', data=df_melt,
              order=order, color=".3", size=2.8, alpha=0.35, jitter=0.12)

plt.title('Distribution of Values Across Networks', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Networks')
plt.ylabel('Value')
plt.tight_layout()

plt.savefig('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_age/INT_40/Step2_subtype/Step4_NetworkHierarchy/subtype2/subtype2_network_boxplot.png', dpi=300, bbox_inches='tight')
print("简单箱体图已保存为: HC_network_boxplot.png")


# ====================== 6. 显著性热力图 (FDR p-value) ======================
# 创建 p-value 矩阵
p_matrix = pd.DataFrame(np.nan, index=networks, columns=networks)

for _, row in posthoc_df.iterrows():
    g1, g2 = row['group1'], row['group2']
    p_matrix.loc[g1, g2] = row['p_fdr']
    p_matrix.loc[g2, g1] = row['p_fdr']

# 对角线设为 1.000（网络自身对比的值）
np.fill_diagonal(p_matrix.values, 1.0)

# 直接按照原始网络的顺序排列行和列
p_matrix = p_matrix.loc[networks, networks]

# 加入 k=1，只隐藏对角线右上方的区域，保留对角线本身
mask = np.triu(np.ones_like(p_matrix, dtype=bool), k=1)

# 绘图
plt.figure(figsize=(12, 10))
sns.heatmap(p_matrix,
            mask=mask,                  # 应用掩码
            annot=True,
            fmt='.3f',
            cmap='viridis_r',           # 颜色越深p值越小（越显著）
            linewidths=0.5,
            cbar_kws={'label': 'FDR corrected p-value'})

plt.title('Pairwise Comparisons (FDR corrected p-values)\nDarker = more significant',
          fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_age/INT_40/Step2_subtype/Step4_NetworkHierarchy/subtype2/subtype2_posthoc_heatmap_fdr.png', dpi=300, bbox_inches='tight')
print("含对角线的下三角热力图已保存为: HC_posthoc_heatmap_fdr.png")