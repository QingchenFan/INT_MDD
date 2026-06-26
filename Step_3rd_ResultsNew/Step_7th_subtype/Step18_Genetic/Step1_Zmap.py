import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# 1. 数据加载与准备
mdd_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype1_INT_agesex_FD.csv')
hc_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step2_Ttest/HC_INT20_agesex_FD.csv')
mdd_df['group'] = 1
hc_df['group'] = 0
combined_df = pd.concat([mdd_df, hc_df], ignore_index=True)

# 2. 定义回归模型
covariates = ['group', 'age', 'sex', 'mean_fd']
X = combined_df[covariates]
X = sm.add_constant(X)
feature_cols = [c for c in combined_df.columns if c not in ['subID', 'mean_fd', 'age', 'sex', 'group']]

# 3. 计算统计量
results = []
for col in feature_cols:
    y = combined_df[col]
    model = sm.OLS(y, X).fit()
    z_stat = model.tvalues['group']
    p_val = model.pvalues['group']
    results.append({'region': col, 'z_stat': z_stat, 'p_val': p_val})

stats_df = pd.DataFrame(results)

# 4. FDR 多重比较校正 (计算 q 值作为参考，不再用于强制数据抹零)
_, q_vals, _, _ = multipletests(stats_df['p_val'], method='fdr_bh')
stats_df['q_val'] = q_vals

# 5. 数据分流与导出：分别保存用于“统计计算”和“可视化”的数据

# ---> 【重点】5.1 导出用于 PLS 基因关联分析的数据 (Unthresholded) <---
# 完全保留 246 个脑区原始连绵起伏的 Z 值，确保空间连续协方差不被破坏
pls_export_df = stats_df[['region', 'z_stat']]
pls_export_df.to_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step18_Genetic/result1_zmap/subtype1_z_map.csv', index=False)

