import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection
import numpy as np

# ===================== 1. 读取数据 =====================
df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step3_Structure/HC_INT_GMV_7net_agesex.csv')

# ===================== 2. 定义网络 =====================
networks = ['subcortical', 'Visual', 'Somatomotor', 'Dorsal_Attention',
            'Ventral_Attention', 'Limbic', 'Frontoparietal', 'Default']

# ===================== 3. 计算相关性 + FDR 校正 =====================
results = []
p_values = []

for net in networks:
    int_col = net
    gmv_col = f'{net}_GMV'
    valid = df[[int_col, gmv_col]].dropna()
    r, p = pearsonr(valid[int_col], valid[gmv_col])

    results.append({
        'network': net,
        'r': round(r, 4),
        'p_raw': round(p, 6),
        'n': len(valid)
    })
    p_values.append(p)

# FDR 校正
_, p_fdr = fdrcorrection(p_values, alpha=0.05)

# 把 FDR 加回结果
for i in range(len(results)):
    results[i]['p_fdr'] = round(p_fdr[i], 6)

# ===================== 4. 打印结果 =====================
print("=== INT-GMV 相关性结果 ===")
for res in results:
    print(f"{res['network']:18} r = {res['r']:.4f}, p = {res['p_raw']:.4f}, p_FDR = {res['p_fdr']:.4f}, n = {res['n']}")

# ===================== 5. 保存结果到 CSV =====================
result_df = pd.DataFrame(results)
result_df = result_df[['network', 'r', 'p_raw', 'p_fdr', 'n']]
result_df.to_csv('INT_GMV_correlation_results.csv', index=False, encoding='utf-8-sig')


# ===================== 6. 绘图 =====================
fig, axes = plt.subplots(2, 4, figsize=(16, 9))
axes = axes.flatten()

for i, net in enumerate(networks):
    ax = axes[i]
    int_col = net
    gmv_col = f'{net}_GMV'

    sns.regplot(x=df[int_col], y=df[gmv_col], ax=ax,
                scatter_kws={'alpha': 0.6, 's': 25},
                line_kws={'color': 'red', 'linewidth': 2})

    res = results[i]
    ax.text(0.05, 0.95,
            f'r = {res["r"]:.3f}\np = {res["p_raw"]:.3f}\nFDR q = {res["p_fdr"]:.3f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel(f'{net} INT', fontsize=11)
    ax.set_ylabel(f'{net} GMV', fontsize=11)
    ax.set_title(net, fontsize=12, fontweight='bold')

plt.suptitle('INT vs GMV Correlation Across 8 Networks', fontsize=16, y=1.02)
plt.tight_layout()

# ===================== 7. 保存图片（已删除 plt.show()） =====================
plt.savefig('HC_INT_GMV_Correlation_8net.png', dpi=300, bbox_inches='tight')
plt.close()  # 关闭图形释放内存
