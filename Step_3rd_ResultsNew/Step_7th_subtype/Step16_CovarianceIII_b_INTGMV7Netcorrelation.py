import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import fdrcorrection

# ==========================================
# 1. 设置路径与加载数据
# ==========================================
base_dir = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationGMV/result1_INTGMV7NetCorrelation'

# 加载 Subtype 1 数据
gmv_r_df = pd.read_csv(f'{base_dir}/subtype1_GMV_covariance_network_r.csv', index_col=0)
gmv_fdr_df = pd.read_csv(f'{base_dir}/subtype1_GMV_covariance_network_fdr.csv', index_col=0)
int_r_df = pd.read_csv(f'{base_dir}/subtype1_INT_covariance_network_r.csv', index_col=0)
int_fdr_df = pd.read_csv(f'{base_dir}/subtype1_INT_covariance_network_fdr.csv', index_col=0)

# 加载分区模板
yeo_df = pd.read_csv('/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/region246_network_Yeo.csv')

# ==========================================
# 2. 构建 8 网络映射 (Yeo 7 + Subcortical)
# ==========================================
yeo8_dict = {
    1: 'Visual', 2: 'Somatomotor', 3: 'Dorsal Attention', 4: 'Ventral Attention',
    5: 'Limbic', 6: 'Frontoparietal', 7: 'Default', 0: 'Subcortical'
}

yeo_df['Yeo_7network'] = yeo_df['Yeo_7network'].fillna(0).astype(int)
valid_yeo = yeo_df[yeo_df['Yeo_7network'].isin(yeo8_dict.keys())].copy()
regions_in_mat = gmv_r_df.columns.tolist()

# ==========================================
# 3. 遍历网络计算并执行 FDR 校正
# ==========================================
results = []
plot_data = {}

for net_id, net_name in yeo8_dict.items():
    net_nodes = valid_yeo[valid_yeo['Yeo_7network'] == net_id]['regions'].tolist()
    net_nodes = [node for node in net_nodes if node in regions_in_mat]

    if len(net_nodes) < 2: continue

    # 提取子矩阵
    sub_gmv_r = gmv_r_df.loc[net_nodes, net_nodes].values
    sub_gmv_fdr = gmv_fdr_df.loc[net_nodes, net_nodes].values
    sub_int_r = int_r_df.loc[net_nodes, net_nodes].values
    sub_int_fdr = int_fdr_df.loc[net_nodes, net_nodes].values

    upper_tri = np.triu_indices(len(net_nodes), k=1)

    gmv_z = np.arctanh(np.clip(sub_gmv_r[upper_tri], -0.9999, 0.9999))
    int_z = np.arctanh(np.clip(sub_int_r[upper_tri], -0.9999, 0.9999))

    # 保留显著边
    sig_mask = (sub_gmv_fdr[upper_tri] < 0.05) | (sub_int_fdr[upper_tri] < 0.05)

    if len(gmv_z[sig_mask]) >= 3:
        r_val, p_val_raw = stats.pearsonr(gmv_z[sig_mask], int_z[sig_mask])
        results.append({
            'Network': net_name, 'Pearson_r': r_val, 'P_value_Raw': p_val_raw,
            'gmv_vec': gmv_z[sig_mask], 'int_vec': int_z[sig_mask]
        })

# FDR 校正
raw_p = [res['P_value_Raw'] for res in results]
_, fdr_p = fdrcorrection(raw_p)

# 填充校正后数据
for i in range(len(results)):
    results[i]['P_value_FDR'] = fdr_p[i]
    plot_data[results[i]['Network']] = (results[i]['gmv_vec'], results[i]['int_vec'], results[i]['Pearson_r'], fdr_p[i])

res_df = pd.DataFrame(results).drop(columns=['gmv_vec', 'int_vec'])
print("\n=== 模块化共变耦合分析结果 (FDR校正后) ===")
print(res_df.to_string(index=False))
res_df.to_csv(f'{base_dir}/Modular_Coupling_Results_FDR.csv', index=False)

# ==========================================
# 4. 可视化
# ==========================================
fig, axes = plt.subplots(2, 4, figsize=(18, 10), facecolor='white')
axes = axes.flatten()
order = ['Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Frontoparietal', 'Default',
         'Subcortical']

for i, net_name in enumerate(order):
    ax = axes[i]
    if net_name in plot_data:
        gmv_v, int_v, r, p_fdr = plot_data[net_name]
        color = '#d62728'#'#d62728' if net_name == 'Subcortical' else '#2ca02c'
        sns.regplot(x=gmv_v, y=int_v, ax=ax, scatter_kws={'alpha': 0.3, 's': 15, 'color': color})
        p_str = "p_FDR < 0.001" if p_fdr < 0.001 else f"p_FDR = {p_fdr:.3f}"
        ax.text(0.05, 0.95, f'r={r:.3f}\n{p_str}', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    ax.set_title(net_name, fontweight='bold')

plt.suptitle('Modular Connectome Coupling (GMV vs INT): Significant Intra-network Edges', fontsize=18)
plt.tight_layout()
plt.savefig(f'{base_dir}/Modular_Coupling_FDR.png', dpi=300)