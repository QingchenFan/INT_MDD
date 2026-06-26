import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.stats.multitest as smm
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
df = pd.read_csv("/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step11_INT_QIDS_Correlation/subtype2_7NetINT_QIDS.csv")

# 2. 定义网络列名和评估指标列名
networks = ['subcortical', 'Visual', 'Somatomotor', 'Dorsal_Attention',
            'Ventral_Attention', 'Limbic', 'Frontoparietal', 'Default']
hamd_col = 'QIDS'

# 3. 计算皮尔逊相关系数和未校正的p值
results = []
for net in networks:
    r, p = pearsonr(df[hamd_col], df[net])
    results.append({'Network': net, 'r': r, 'p': p})

results_df = pd.DataFrame(results)

# 4. 使用 FDR (Benjamini-Hochberg) 方法进行多重比较校正
rej, pval_corr, _, _ = smm.multipletests(results_df['p'], alpha=0.05, method='fdr_bh')
results_df['p_fdr'] = pval_corr

# 5. 绘制散点图
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, net in enumerate(networks):
    ax = axes[i]
    # 使用 seaborn 绘制散点图及回归线
    sns.regplot(data=df, x=hamd_col, y=net, ax=ax, scatter_kws={'alpha': 0.6})
    r_val = results_df.loc[i, 'r']
    p_fdr = results_df.loc[i, 'p_fdr']
    p_uncorr = results_df.loc[i, 'p']

    # 在子图中添加文本框显示相关系数和p值
    ax.set_title(f"{net}")
    ax.text(0.05, 0.95, f"r = {r_val:.3f}\np_fdr = {p_fdr:.3f}\n(p = {p_uncorr:.3f})",
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('s2_QIDS_networks_correlation.png', dpi=300)
plt.show()