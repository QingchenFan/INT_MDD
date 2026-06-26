# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# '''
#     共变网络的边，两个亚型做差值，然后映射到网络水平。
# '''
# # ==========================================
# # 1. 加载数据
# # ==========================================
# print("正在加载数据...")
# df1 = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_Covariation/subtype1_INTcovariance_network.csv', index_col=0)
# df2 = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_Covariation/subtype2_INTcovariance_network.csv', index_col=0)
# yeo_df = pd.read_csv('/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/region246_network_Yeo.csv')
#
# # ==========================================
# # 2. 计算差异矩阵 (Subtype 1 - Subtype 2)
# # ==========================================
# delta_mat = df1 - df2
#
# # ==========================================
# # 3. 根据所提供的文件构建映射字典
# # ==========================================
# yeo_dict = {
#     1: 'Visual',
#     2: 'Somatomotor',
#     3: 'Dorsal Attention',
#     4: 'Ventral Attention',
#     5: 'Limbic',
#     6: 'Frontoparietal',
#     7: 'Default'
# }
#
# thalamus_regions = [
#     'mPFtha_L', 'mPFtha_R', 'mPMtha_L', 'mPMtha_R',
#     'Stha_L', 'Stha_R', 'rTtha_L', 'rTtha_R',
#     'PPtha_L', 'PPtha_R', 'Otha_L', 'Otha_R',
#     'cTtha_L', 'cTtha_R', 'lPFtha_L', 'lPFtha_R'
# ]
#
# # 遍历 Yeo 映射表，生成 {脑区: 所属网络} 字典
# mapping = {}
# for index, row in yeo_df.iterrows():
#     # 提取脑区名称，对应 BN246 中的列名
#     region_name = row['regions']
#     yeo_net = row['Yeo_7network']
#
#     # 首先判断是否为丘脑
#     if region_name in thalamus_regions:
#         mapping[region_name] = 'Thalamus'
#     # 其次判断是否有对应的 Yeo 网络 (1-7)
#     elif pd.notna(yeo_net) and yeo_net in yeo_dict:
#         mapping[region_name] = yeo_dict[yeo_net]
#     # 其余未被分类的脑区（比如海马、杏仁核、基底节），归入 Subcortical
#     else:
#         mapping[region_name] = 'Subcortical'
#
# # ==========================================
# # 4. 将映射应用到 246*246 差异矩阵
# # ==========================================
# delta_sys = delta_mat.copy()
# # 替换列名和行名为宏观网络名称
# delta_sys.columns = delta_sys.columns.map(mapping)
# delta_sys.index = delta_sys.index.map(mapping)
#
# # ==========================================
# # 5. 计算宏观网络间的平均差异
# # ==========================================
# # 对行求均值，再对列求均值，最终压缩为宏观网络的均值矩阵
# network_level_delta = delta_sys.groupby(level=0, axis=0).mean().groupby(level=0, axis=1).mean()
#
# # 定义希望在结果中展示的系统排序
# order = ['Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention',
#          'Limbic', 'Frontoparietal', 'Default', 'Subcortical', 'Thalamus']
#
# # 对矩阵进行重新排序
# network_level_delta = network_level_delta.loc[order, order]
#
# print("\n=== 网络水平的 INT 共变重组 (Subtype 1 - Subtype 2) ===")
# print(network_level_delta.round(4).to_string())
#
# # ==========================================
# # 6. 保存计算结果
# # ==========================================
# output_csv = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_Covariation/result2_subtypeEdgDiff/network_level_difference.csv'
# network_level_delta.to_csv(output_csv)
# print(f"\n结果已保存至 {output_csv}")
#
# # 绘制并保存热力图
# plt.figure(figsize=(10, 8))
# sns.heatmap(network_level_delta, cmap='vlag', center=0, annot=True, fmt=".4f",
#             cbar_kws={'label': 'Delta (Subtype 1 - Subtype 2)'})
# plt.title("Network-Level INT Covariance Difference")
# plt.tight_layout()
# plt.savefig('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_Covariation/result2_subtypeEdgDiff/network_reorganization_Yeo7_Thalamus.png', dpi=300)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 加载数据 (请将路径替换为您的实际路径)
# ==========================================
print("正在加载数据...")

df1 = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_Covariation/subtype1_INTcovariance_network.csv', index_col=0)
df2 = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_Covariation/subtype2_INTcovariance_network.csv', index_col=0)
yeo_df = pd.read_csv('/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/region246_network_Yeo.csv')

# ==========================================
# 2. 计算差异矩阵: Delta = Subtype1 - Subtype2
# ==========================================
delta_mat = df2 - df1

# ==========================================
# 3. 构建 Yeo 17 映射字典
# ==========================================
yeo17_dict = {
    1: 'VisCent',
    2: 'VisPeri',
    3: 'SomMotA',
    4: 'SomMotB',
    5: 'DorsAttnA',
    6: 'DorsAttnB',
    7: 'SalVentAttnA',
    8: 'SalVentAttnB',
    9: 'LimbicA',
    10: 'LimbicB',
    11: 'ContA',
    12: 'ContB',
    13: 'ContC',
    14: 'DefaultA',
    15: 'DefaultB',
    16: 'DefaultC',
    17: 'TempPar'
}

thalamus_regions = [
    'mPFtha_L', 'mPFtha_R', 'mPMtha_L', 'mPMtha_R',
    'Stha_L', 'Stha_R', 'rTtha_L', 'rTtha_R',
    'PPtha_L', 'PPtha_R', 'Otha_L', 'Otha_R',
    'cTtha_L', 'cTtha_R', 'lPFtha_L', 'lPFtha_R'
]

mapping = {}
for index, row in yeo_df.iterrows():
    region_name = row['regions']
    yeo_net = row['Yeo_17network']

    if region_name in thalamus_regions:
        mapping[region_name] = 'Thalamus'
    elif pd.notna(yeo_net) and yeo_net in yeo17_dict:
        mapping[region_name] = yeo17_dict[yeo_net]
    else:
        mapping[region_name] = 'Drop'

# ==========================================
# 4. 映射、降维并剔除不需要的脑区
# ==========================================
delta_sys = delta_mat.copy()
delta_sys.columns = delta_sys.columns.map(mapping)
delta_sys.index = delta_sys.index.map(mapping)

if 'Drop' in delta_sys.columns:
    delta_sys = delta_sys.drop('Drop', axis=1)
if 'Drop' in delta_sys.index:
    delta_sys = delta_sys.drop('Drop', axis=0)

network_level_delta = delta_sys.groupby(level=0, axis=0).mean().groupby(level=0, axis=1).mean()

order = [
    'VisCent', 'VisPeri', 'SomMotA', 'SomMotB',
    'DorsAttnA', 'DorsAttnB', 'SalVentAttnA', 'SalVentAttnB',
    'LimbicA', 'LimbicB', 'ContA', 'ContB', 'ContC',
    'DefaultA', 'DefaultB', 'DefaultC', 'TempPar', 'Thalamus'
]

existing_order = [net for net in order if net in network_level_delta.columns]
network_level_delta = network_level_delta.loc[existing_order, existing_order]

print("\n=== Yeo17 + Thalamus 宏观网络共变差异 ===")
print(network_level_delta.round(4).to_string())

# ==========================================
# 5. 保存结果及热力图可视化 (已修改)
# ==========================================
network_level_delta.to_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_Covariation/result2_subtypeEdgDiff/network_reorganization_yeo17.csv')

plt.figure(figsize=(16, 14)) # 稍微加大了一点画布，给数字留足空间

# ---> 修改点在这里：annot=True 显示数字，fmt=".2f" 保留两位小数，annot_kws 调整字号
sns.heatmap(network_level_delta, cmap='vlag', center=0,
            annot=True, fmt=".2f", annot_kws={"size": 8, "weight": "bold"},
            cbar_kws={'label': 'Delta (Subtype 2 - Subtype 1)'}, square=True)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.title("Network-Level INT Covariance Difference (Yeo 17 + Thalamus)", fontsize=16, pad=20)
plt.tight_layout()

# 保存图片
plt.savefig('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_Covariation/result2_subtypeEdgDiff/network_reorganization_yeo17.png', dpi=300)
print("\nCSV 与热力图 PNG 文件已生成！")