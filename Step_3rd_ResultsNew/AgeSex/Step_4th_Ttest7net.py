import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import seaborn as sns

# ================== 1. 读取数据 ==================
hc_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step3_Structure/HC_GMV_7Net_agesex.csv')
mdd_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step3_Structure/MDD_GMV_7Net_agesex.csv')

print(f"HC 被试数: {len(hc_df)}, MDD 被试数: {len(mdd_df)}")

# 网络列名（去掉 subID）
networks = ['subcortical', 'Visual', 'Somatomotor', 'Dorsal_Attention',
            'Ventral_Attention', 'Limbic', 'Frontoparietal', 'Default']
# networks = ['subcortical_GMV', 'Visual_GMV', 'Somatomotor_GMV', 'Dorsal_Attention_GMV',
#             'Ventral_Attention_GMV', 'Limbic_GMV', 'Frontoparietal_GMV', 'Default_GMV']
# ================== 2. 双样本 t 检验 + FDR 校正 ==================
results = []

for net in networks:
    hc_vals = hc_df[net].dropna()
    mdd_vals = mdd_df[net].dropna()

    # Welch's t-test (不假设方差相等)
    t_stat, p_val = stats.ttest_ind(hc_vals, mdd_vals, equal_var=False, alternative='two-sided')

    # 均值和标准差
    hc_mean = hc_vals.mean()
    mdd_mean = mdd_vals.mean()
    hc_std = hc_vals.std()
    mdd_std = mdd_vals.std()

    # Cohen's d 效应量
    pooled_std = np.sqrt(
        ((len(hc_vals) - 1) * hc_std ** 2 + (len(mdd_vals) - 1) * mdd_std ** 2) / (len(hc_vals) + len(mdd_vals) - 2))
    cohen_d = (hc_mean - mdd_mean) / pooled_std if pooled_std != 0 else np.nan

    results.append({
        'Network': net,
        'HC_mean': round(hc_mean, 4),
        'HC_std': round(hc_std, 4),
        'MDD_mean': round(mdd_mean, 4),
        'MDD_std': round(mdd_std, 4),
        'Diff (HC-MDD)': round(hc_mean - mdd_mean, 4),
        't': round(t_stat, 3),
        'p_raw': p_val,
        'Cohen_d': round(cohen_d, 3)
    })

# 转为 DataFrame
result_df = pd.DataFrame(results)

# FDR 校正
p_raw = result_df['p_raw'].values
_, p_fdr = fdrcorrection(p_raw, alpha=0.05, method='indep')
result_df['p_fdr'] = p_fdr
result_df['significant'] = result_df['p_fdr'] < 0.05

# 排序显示（按 p_fdr）
result_df = result_df.sort_values('p_fdr')

print("\n=== HC vs MDD 网络层面 t 检验结果 (FDR校正) ===")
print(result_df.round(4))

# 保存表格
result_df.to_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step3_Structure/Ttest_MDDHC_GMV7net.csv', index=False)


# ================== 3. 绘制一张画布上的 8 个网络箱线图 ==================
plt.figure(figsize=(16, 10))

# 准备长格式数据用于 seaborn
plot_data = []
for net in networks:
    for val, group in zip(hc_df[net], ['HC'] * len(hc_df)):
        plot_data.append({'Network': net, 'Group': 'HC', 'INT': val})
    for val, group in zip(mdd_df[net], ['MDD'] * len(mdd_df)):
        plot_data.append({'Network': net, 'Group': 'MDD', 'INT': val})

plot_df = pd.DataFrame(plot_data)

# 绘图
ax = sns.boxplot(x='Network', y='INT', hue='Group', data=plot_df,
                 palette={'HC': '#1f77b4', 'MDD': '#d62728'},
                 width=0.6, fliersize=3)

# 添加散点
sns.stripplot(x='Network', y='INT', hue='Group', data=plot_df,
              dodge=True, alpha=0.6, jitter=True, size=4,
              palette={'HC': '#1f77b4', 'MDD': '#d62728'})

# 添加显著性标记
for i, net in enumerate(networks):
    p = result_df.loc[result_df['Network'] == net, 'p_fdr'].values[0]
    if p < 0.001:
        sig = '***'
    elif p < 0.01:
        sig = '**'
    elif p < 0.05:
        sig = '*'
    else:
        sig = 'n.s.'

    # 在箱线图上方添加标记
    y_max = plot_df[plot_df['Network'] == net]['INT'].max() * 1.05
    plt.text(i, y_max, sig, ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.title('INT Values Comparison between HC and MDD across 8 Yeo Networks', fontsize=16, pad=20)
plt.xlabel('Yeo 7 Networks + Subcortical', fontsize=14)
plt.ylabel('Integrated Network Threshold (INT)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Group', fontsize=12)

# 美化
plt.grid(axis='y', alpha=0.3)
sns.despine()

plt.tight_layout()
plt.savefig('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step3_Structure/Ttest_MDDHC_GMV7net.png', dpi=300, bbox_inches='tight')
#plt.show()

