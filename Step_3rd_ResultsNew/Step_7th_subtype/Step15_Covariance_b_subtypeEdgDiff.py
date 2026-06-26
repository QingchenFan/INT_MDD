import pandas as pd
import numpy as np

# ==========================================
# 0. 数据加载
# ==========================================
print("正在加载数据...")
# 请替换为您本地的实际文件路径
df1 = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_Covariation/subtype1_INTcovariance_network.csv', index_col=0)
df2 = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_Covariation/subtype2_INTcovariance_network.csv', index_col=0)

# 计算差值矩阵: Delta = Subtype1 - Subtype2
delta_mat = df1 - df2

# ==========================================
# 第一部分：边水平分析 (Edge-level Analysis)
# ==========================================
print("\n=== 第一部分：计算最大差异边并保存至CSV ===")

# 提取矩阵的上三角索引（k=1 排除对角线）
row_idx, col_idx = np.triu_indices(len(df1.columns), k=1)

# 构建边的 DataFrame
edges = pd.DataFrame({
    'Region_1': df1.columns[row_idx],
    'Region_2': df1.columns[col_idx],
    'Delta': delta_mat.values[row_idx, col_idx]
})

# 计算绝对差异
edges['Abs_Delta'] = edges['Delta'].abs()

# ---> 修改点1：将两个脑区名称用 "-" 连接起来作为第一列
edges['Edge_Connection'] = edges['Region_1'] + '-' + edges['Region_2']

# ---> 修改点2：只保留我们需要的两列，并按绝对差值降序排序（最大的排在最前面）
result_edges = edges[['Edge_Connection', 'Abs_Delta']].sort_values(by='Abs_Delta', ascending=False)

# ---> 修改点3：将结果保存到 CSV 文件中
output_csv_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_Covariation/result2_subtypeNetDiff/edge_absolute_differences.csv'
result_edges.to_csv(output_csv_path, index=False)

print(f"数据已成功保存至 {output_csv_path}")
print("前 5 个绝对差值最大的连接如下：")
print(result_edges.head().to_string(index=False))

# （如果您还需要运行第二部分节点水平的代码，可以紧接着附在下方）