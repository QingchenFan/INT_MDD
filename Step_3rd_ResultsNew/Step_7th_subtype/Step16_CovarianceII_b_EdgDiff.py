import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# ---------------- 1. 参数设置 ----------------
# 【重要】请输入两组实际的样本量！
n1 = 156  # subtype1 的样本量 (请核实)
n2 = 179  # subtype2 的样本量 (请替换为实际人数!)

# ---------------- 2. 加载数据 ----------------
# 加载 R 值矩阵
r1_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationII/subtype1_INT_covariance_network_r.csv', index_col=0)
r2_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationII/subtype2_INT_covariance_network_r.csv', index_col=0)

# 加载组内 FDR 显著性矩阵
p1_fdr_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationII/subtype1_INT_covariance_network_fdr.csv', index_col=0)
p2_fdr_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationII/subtype2_INT_covariance_network_fdr.csv', index_col=0)

regions = r1_df.columns
n_regions = len(regions)

# ---------------- 3. 确定需要检验的边 ----------------
# 逻辑：至少在一个组中 FDR < 0.05 的边，才被纳入组间比较的候选名单
sig_mask = (p1_fdr_df.values < 0.05) | (p2_fdr_df.values < 0.05)

# ---------------- 4. Fisher's r-to-z 转换 ----------------
# 限制 R 值的范围，防止出现对角线 R=1.0 时计算 arctanh 报错 (除以0)
r1_clipped = np.clip(r1_df.values, -0.999999, 0.999999)
r2_clipped = np.clip(r2_df.values, -0.999999, 0.999999)

z1 = np.arctanh(r1_clipped)
z2 = np.arctanh(r2_clipped)

# ---------------- 5. 计算组间差异 Z 统计量 ----------------
# 根据文献公式计算标准误 SE
se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))

# 计算组间差异的 Z 值
z_diff = (z1 - z2) / se

# 转换为双侧检验的 P 值 (基于标准正态分布)
p_diff = 2 * stats.norm.sf(np.abs(z_diff))

# ---------------- 6. FDR 多重比较校正 ----------------
# 初始化组间差异的 FDR P 值矩阵 (默认置为1)
fdr_diff = np.ones_like(p_diff)

# 只提取上三角部分 (不包含对角线)，避免重复校正
upper_tri = np.triu_indices(n_regions, k=1)

# 获取上三角中需要做检验的边 (布尔索引)
mask_upper = sig_mask[upper_tri]

# 提取真正需要检验的 p 值
p_to_test = p_diff[upper_tri][mask_upper]

# 执行 FDR 校正 (如果有显著边的话)
if len(p_to_test) > 0:
    rejected, p_fdr_corrected = fdrcorrection(p_to_test, alpha=0.05, method='indep')

    # 构建一个全是 1 的上三角一维数组，并将经过校正的 p 值放回原位置
    p_fdr_upper = np.ones(len(upper_tri[0]))
    p_fdr_upper[mask_upper] = p_fdr_corrected

    # 填回 2D 矩阵的上三角区域
    fdr_diff[upper_tri] = p_fdr_upper

    # 镜像填充到左下三角区域，使矩阵对称
    lower_tri = np.tril_indices(n_regions, k=-1)
    fdr_diff[lower_tri] = fdr_diff.T[lower_tri]

# ---------------- 7. 保存结果 ----------------
# 将差异 Z 矩阵、差异 P 值矩阵、校正后 FDR 矩阵转换为 DataFrame
Z_diff_df = pd.DataFrame(z_diff, index=regions, columns=regions)
P_diff_df = pd.DataFrame(p_diff, index=regions, columns=regions)
FDR_diff_df = pd.DataFrame(fdr_diff, index=regions, columns=regions)

# 组间差异方向处理: 为了后续画图方便，可以在不显著的地方置 0
# 生成一个“显著差异网络”矩阵 (有差异保留 Z 值，无差异置 0)
Sig_Network_df = Z_diff_df.copy()
Sig_Network_df[FDR_diff_df >= 0.05] = 0

Z_diff_df.to_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationII/result1_EdgDiff/BetweenGroup_Z_diff.csv')
P_diff_df.to_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationII/result1_EdgDiff/BetweenGroup_P_diff.csv')
FDR_diff_df.to_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationII/result1_EdgDiff/BetweenGroup_FDR_diff.csv')
Sig_Network_df.to_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationII/result1_EdgDiff/BetweenGroup_Significant_Network.csv')

print("组间比较计算完成，结果已保存为 CSV 文件。")
# ---------------- 提取并排序显著差异边 ----------------
# 仅提取上三角部分以避免重复 (A-B 和 B-A)
mask_edges = np.triu(np.ones(Sig_Network_df.shape), k=1).astype(bool)

row_idx, col_idx = np.where(mask_edges)
z_values_edges = Sig_Network_df.values[mask_edges]

edges = []
for r, c, z in zip(row_idx, col_idx, z_values_edges):
    if z != 0: # 剔除不显著的 0 值
        edges.append({
            'Node_1': regions[r],
            'Node_2': regions[c],
            'Z_value': z,
            'Abs_Z_value': abs(z)
        })

edges_df = pd.DataFrame(edges)

if not edges_df.empty:
    edges_df = edges_df.sort_values(by='Abs_Z_value', ascending=False).reset_index(drop=True)
    edges_save_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationII/result1_EdgDiff/Sorted_Significant_Edges.csv'
    edges_df.to_csv(edges_save_path, index=False)
    print(f"提取排序完成！共找到 {len(edges_df)} 条显著组间差异连接，已保存至 Sorted_Significant_Edges.csv。")
else:
    print("提取排序完成：未发现显著的组间差异连线。")

# 1. 加载显著性差异网络矩阵
df = Sig_Network_df

# 2. 设置画布大小
plt.figure(figsize=(12, 10), facecolor='white')

# 3. 确定颜色映射的范围，使 0 值严格居中对齐对应的中性色（白色）
max_val = np.abs(df.values).max()
if max_val == 0:
    max_val = 1 # 防止全是 0 报错的情况

# ----------------- 新增部分：创建上三角掩膜 -----------------
# np.ones_like 生成全为 True 的矩阵，np.triu 提取上三角部分（包含对角线）
# 这样就可以把上三角和全是 0 的对角线遮挡掉，只留下纯粹的左下三角
mask = np.triu(np.ones_like(df, dtype=bool))
# ------------------------------------------------------------

# 4. 绘制热力图
# 加入 mask=mask 参数
sns.heatmap(df,
            mask=mask,
            cmap="RdBu_r",
            center=0,
            vmin=-max_val,
            vmax=max_val,
            square=True,
            xticklabels=False,  # 246个脑区名字太多，隐藏坐标轴具体标签以保持整洁
            yticklabels=False,
            cbar_kws={"shrink": .8, "label": "Z-value (Subtype1 - Subtype2)"})

# 5. 添加标题和标签
plt.title('Significant Structural Covariance Differences\n(Subtype1 vs Subtype2)', fontsize=16, pad=20)
plt.xlabel('246 Brain Regions', fontsize=12)
plt.ylabel('246 Brain Regions', fontsize=12)

# 6. 保存图片
plt.tight_layout()
plt.savefig('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationII/result1_EdgDiff/BetweenGroup_Heatmap.png', facecolor='white',dpi=300, transparent=True)
plt.close()