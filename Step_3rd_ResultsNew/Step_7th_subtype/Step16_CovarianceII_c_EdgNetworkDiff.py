import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 加载数据
# ==========================================

# 【修改点 1】：加载显著 Z 值差异矩阵 (246*246)，而不是原始的 r 值矩阵
delta_mat = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationII/result1_EdgDiff/BetweenGroup_Significant_Network.csv', index_col=0)

# 加载 Yeo 分区模板字典
yeo_df = pd.read_csv('/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/region246_network_Yeo.csv')

# ==========================================
# 2. 构建 Yeo 17 映射字典
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
# 3. 映射、降维并剔除不需要的脑区
# ==========================================
delta_sys = delta_mat.copy()
delta_sys.columns = delta_sys.columns.map(mapping)
delta_sys.index = delta_sys.index.map(mapping)

if 'Drop' in delta_sys.columns:
    delta_sys = delta_sys.drop('Drop', axis=1)
if 'Drop' in delta_sys.index:
    delta_sys = delta_sys.drop('Drop', axis=0)

# 计算宏观网络水平的均值
network_level_delta = delta_sys.groupby(level=0, axis=0).mean().groupby(level=0, axis=1).mean()

order = [
    'VisCent', 'VisPeri', 'SomMotA', 'SomMotB',
    'DorsAttnA', 'DorsAttnB', 'SalVentAttnA', 'SalVentAttnB',
    'LimbicA', 'LimbicB', 'ContA', 'ContB', 'ContC',
    'DefaultA', 'DefaultB', 'DefaultC', 'TempPar', 'Thalamus'
]

existing_order = [net for net in order if net in network_level_delta.columns]
network_level_delta = network_level_delta.loc[existing_order, existing_order]

print("\n=== Yeo17 + Thalamus 宏观网络显著差异 Z 值均值 ===")
print(network_level_delta.round(4).to_string())

# ==========================================
# 4. 保存结果及热力图可视化
# ==========================================
network_level_delta.to_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationII/result1_EdgDiff/network_reorganization_yeo17_Zdiff.csv')

# 设置画布大小，背景色为白
fig = plt.figure(figsize=(16, 14), facecolor='white')

# 【修改点 2】：引入左下三角掩膜 (Mask)
mask = np.triu(np.ones_like(network_level_delta, dtype=bool),k=1)

# 根据降维后的矩阵找最大绝对值，用于对称映射颜色
max_val = np.abs(network_level_delta.values).max()
if max_val == 0:
    max_val = 1

# 绘制带有数字标注的左下三角热力图
ax = sns.heatmap(network_level_delta, mask=mask, cmap='vlag', center=0,
                 vmin=-max_val, vmax=max_val,
                 annot=True, fmt=".3f", annot_kws={"size": 10, "weight": "bold"}, # 显示 3 位小数
                 cbar_kws={'label': 'Mean Z-value Diff (Subtype 1 - Subtype 2)'}, square=True)

# 防露底背景颜色配置
ax.set_facecolor('white')

plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)

# 【修改点 3】：更新图表标题，标明这是显著 Z 值差异图
plt.title("Significant Network-Level Covariance Difference\n(Yeo 17 + Thalamus)", fontsize=18, pad=20)
plt.tight_layout()

# 保存图片
plt.savefig('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationII/result1_EdgDiff/network_reorganization_yeo17_Zdiff2.png', dpi=300, facecolor='white', transparent=False)
plt.close()

print("\nCSV 与左下三角热力图 PNG 文件已生成！")