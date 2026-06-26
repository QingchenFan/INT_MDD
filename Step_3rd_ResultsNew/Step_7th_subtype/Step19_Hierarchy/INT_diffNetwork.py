import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------- 0. 基础路径和名称设置 -----------------
base_dir = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/'
out_dir = os.path.join(base_dir, 'Step1_NetworkHierarchy')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)  # 如果输出文件夹不存在则创建

file1 = os.path.join(base_dir, 'subtype1_INT_7net_agesex_FD.csv')
file2 = os.path.join(base_dir, 'subtype2_INT_7net_agesex_FD.csv')

ordered_networks = [
    'Default', 'Frontoparietal', 'Limbic', 'Ventral_Attention',
    'Dorsal_Attention', 'Somatomotor', 'Visual', 'subcortical'
]

# ----------------- 1. 读取数据并计算 3D 差异矩阵 -----------------
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

data1 = df1[ordered_networks].values
data2 = df2[ordered_networks].values

# 计算 3D 差异矩阵
diff_1 = data1[:, :, np.newaxis] - data1[:, np.newaxis, :]
diff_2 = data2[:, :, np.newaxis] - data2[:, np.newaxis, :]

# ----------------- 2. 提取上三角数据，拉平并与协变量对应保存 -----------------
# 提取上三角索引 (k=1 表示不包含主对角线)，共有 8*7/2 = 28 个连接
upper_tri_indices = np.triu_indices(8, k=1)

# 生成这 28 个连接的名称，例如 "Default-Frontoparietal"
edge_names = [f"{ordered_networks[i]}-{ordered_networks[j]}" for i, j in zip(*upper_tri_indices)]


def flatten_and_merge(df_orig, diff_3d, group_label):
    # 提取所有被试的上三角数据，形状变为 (N, 28)
    flattened_edges = diff_3d[:, upper_tri_indices[0], upper_tri_indices[1]]

    # 转换为 DataFrame
    df_edges = pd.DataFrame(flattened_edges, columns=edge_names)

    # 提取协变量
    df_covariates = df_orig[['subID', 'age', 'sex', 'mean_fd']].copy()
    df_covariates['Group'] = group_label  # 添加组别标签 (1代表Subtype1, 0代表Subtype2)

    # 拼接并返回
    return pd.concat([df_covariates, df_edges], axis=1)


# 处理两组数据并保存
df1_flat = flatten_and_merge(df1, diff_1, group_label=1)
df2_flat = flatten_and_merge(df2, diff_2, group_label=0)

df1_flat.to_csv(os.path.join(out_dir, 'subtype1_edges_flattened.csv'), index=False)
df2_flat.to_csv(os.path.join(out_dir, 'subtype2_edges_flattened.csv'), index=False)
print("需求1已完成：单被试的 28 个网络差异已拉平并与协变量合并保存。")

# ----------------- 3. 控制协变量的组间比较 (OLS回归) -----------------
# 合并两组数据用于统计分析
df_all = pd.concat([df1_flat, df2_flat], ignore_index=True)

# 定义自变量 X (组别 + 年龄 + 性别 + meanFD)
# 注意：我们要看的是 Group 的主效应，Subtype1(1) vs Subtype2(0) 意味着 T为正则 Subtype1的差值更大
X = df_all[['Group', 'age', 'sex', 'mean_fd']]
X = sm.add_constant(X)  # 添加常数项(截距)

t_vals_28 = []
p_vals_28 = []

# 对 28 个连接逐一进行多元线性回归
for edge in edge_names:
    y = df_all[edge]
    model = sm.OLS(y, X).fit()
    # 提取 'Group' 这一项的 t值 和 p值
    t_vals_28.append(model.tvalues['Group'])
    p_vals_28.append(model.pvalues['Group'])

# ----------------- 4. p值校正与 8x8 矩阵还原 -----------------
# 使用 Benjamini/Hochberg 方法对 28 个 P 值进行 FDR 校正
reject, p_vals_corr_28, _, _ = multipletests(p_vals_28, alpha=0.05, method='fdr_bh')

# 初始化 8x8 全零(或全一)矩阵
t_stat_matrix = np.zeros((8, 8))
p_val_corrected_matrix = np.ones((8, 8))

# 将 28 个结果填回上三角
t_stat_matrix[upper_tri_indices] = t_vals_28
p_val_corrected_matrix[upper_tri_indices] = p_vals_corr_28

# 对称地填补至左下角（注意 T 值 A-B 等于 -(B-A)，而 P 值是对称相等的）
# 右上角存的是 row-col，左下角应该是 col-row = -(row-col)
t_stat_matrix.T[upper_tri_indices] = [-t for t in t_vals_28]
p_val_corrected_matrix.T[upper_tri_indices] = p_vals_corr_28

# 保存为 DataFrame
t_map_df = pd.DataFrame(t_stat_matrix, index=ordered_networks, columns=ordered_networks)
p_map_df = pd.DataFrame(p_val_corrected_matrix, index=ordered_networks, columns=ordered_networks)

t_map_df.to_csv(os.path.join(out_dir, 't_map_matrix_covariates.csv'))
p_map_df.to_csv(os.path.join(out_dir, 'p_map_corrected_matrix_covariates.csv'))
print("需求2已完成：控制年龄、性别、meanFD的组间比较已完成并保存。")

# ----------------- 5. 绘制 T-map 热图 -----------------
# 为了让热图更好看，我们创建一个带 '*' 的文本标注矩阵，p_corr < 0.05 的位置打星号
annot_matrix = np.empty((8, 8), dtype=object)
for i in range(8):
    for j in range(8):
        if i == j:
            annot_matrix[i, j] = ""  # 对角线留白
        else:
            t_val = t_stat_matrix[i, j]
            p_val = p_val_corrected_matrix[i, j]
            stars = "*" if p_val < 0.05 else ""  # 如果需要区分 0.01 也可以加 "**"
            # 格式：显示 T 值，如果显著再加个星号
            annot_matrix[i, j] = f"{t_val:.2f}{stars}"

plt.figure(figsize=(10, 8))
# 绘制热图，中心设为0，蓝色代表负值，红色代表正值
ax = sns.heatmap(t_map_df, cmap='coolwarm', center=0,
                 annot=annot_matrix, fmt='',
                 square=True, linewidths=.5,
                 cbar_kws={"label": "T-value (Subtype1 vs Subtype2)"})

plt.title('T-map of Network Differences\n(Controlling for Age, Sex, meanFD; * FDR $p<0.05$)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# 保存热图
heatmap_path = os.path.join(out_dir, 't_map_heatmap.png')
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"需求3已完成：热图已保存至 {heatmap_path}")
plt.show()