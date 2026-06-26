import pandas as pd
import numpy as np

# ==========================================
# 0. 数据加载
# ==========================================
print("正在加载数据...")
# 使用您提供的文件路径
input_path_sub1 = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_Covariation/subtype1_INTcovariance_network.csv'
input_path_sub2 = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_Covariation/subtype2_INTcovariance_network.csv'

df1 = pd.read_csv(input_path_sub1, index_col=0)
df2 = pd.read_csv(input_path_sub2, index_col=0)

# ==========================================
# 第二部分：节点水平分析 (Node Strength Analysis)
# ==========================================
print("\n=== 第二部分：计算节点强度差异并保存至CSV ===")

# 1. 计算每个节点的 Strength
# 将当前脑区所在行的所有连边权重相加，即为该节点的 Strength
strength1 = df1.sum(axis=1)
strength2 = df2.sum(axis=1)

# 2. 计算节点强度的差值: Delta = Subtype1 - Subtype2
delta_strength = strength1 - strength2

# 3. 构建节点的 DataFrame
nodes_df = pd.DataFrame({
    'Brain_Region': df1.index,          # 脑区名称
    'Subtype1_Strength': strength1.values, # 亚型1的节点强度
    'Subtype2_Strength': strength2.values, # 亚型2的节点强度
    'Delta_Strength': delta_strength.values # 差值
})

# 4. 计算绝对差异并按绝对差异降序排序（变化最大的排在最前）
nodes_df['Abs_Delta_Strength'] = nodes_df['Delta_Strength'].abs()
nodes_df = nodes_df.sort_values(by='Abs_Delta_Strength', ascending=False)

# 5. 保存结果为 CSV 文件
output_csv_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_Covariation/result3_subtypeStrengthDiff/node_strength_differences.csv'
nodes_df.to_csv(output_csv_path, index=False)

print(f"节点 Strength 分析数据已成功保存至：\n{output_csv_path}")
print("\n前 10 个节点强度绝对差值最大的脑区如下：")
# 在控制台打印前 10 行供您预览
print(nodes_df.head(10).to_string(index=False))