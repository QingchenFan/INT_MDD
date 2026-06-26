import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
from nilearn import plotting
import nibabel as nib
from brainsmash.mapgen.base import Base

# ==========================================
# 1. 读取网络与 T-map 数据
# ==========================================
print("正在读取网络和 T-map 数据...")
network_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step16_Epicenters/HC_INT_covariance_network_r.csv', index_col=0)
tmap_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step16_Epicenters/subtype2_t_map_results.csv')

# 确保 T-map 的顺序与网络矩阵的列名完全一致
regions = network_df.columns.tolist()
n_regions = len(regions)  # 这里应该是 246

tmap_df_sorted = tmap_df.set_index('Region').reindex(regions)
t_values = tmap_df_sorted['t_value'].values
network_mat = network_df.values

# ==========================================
# 2. 提取全脑 3D 坐标并计算“距离矩阵”
# ==========================================
print("正在从 BNA246 模板中提取空间坐标...")
atlas_path = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_246_2mm.nii.gz'

# 获取 246 个脑区的中心坐标 (x, y, z)
coords = plotting.find_parcellation_cut_coords(labels_img=atlas_path)

# 计算 246x246 的欧氏距离矩阵 (Euclidean Distance Matrix)
# 这是 BrainSMASH 替代几何旋转的核心基础
print("正在计算全脑空间距离矩阵...")
dist_mat = squareform(pdist(coords, metric='euclidean'))

# ==========================================
# 3. 计算真实的经验震中相关性 (Empirical R)
# ==========================================
print("正在计算经验震中空间相关性...")
epicenter_r = np.zeros(n_regions)

for i in range(n_regions):
    # 提取第 i 个脑区与全脑所有 246 个脑区的连接模式
    profile = network_mat[i, :]
    r, _ = pearsonr(profile, t_values)
    epicenter_r[i] = r

# ==========================================
# 4. 使用 BrainSMASH 生成空间零模型
# ==========================================
n_permutations = 10000
print(f"正在使用 BrainSMASH 生成 {n_permutations} 个空间替代图谱 (Surrogate maps)...")


# 初始化 BrainSMASH，传入原始的 t-map 和计算好的距离矩阵
base = Base(x=t_values, D=dist_mat)
# 生成 10,000 个打乱但保留了空间平滑性的假 t-map
# surrogate_maps 的维度将是 (10000, 246)
surrogate_maps = base(n=n_permutations)

# ==========================================
# 5. 计算 P 值 (p_surrogate)
# ==========================================
print("正在对比零模型，计算显著性 p 值...")
p_surrogate = np.zeros(n_regions)

for i in range(n_regions):
    emp_r = epicenter_r[i]
    profile = network_mat[i, :]
    null_rs = np.zeros(n_permutations)

    # 遍历 10,000 个假 t-map 进行相关性计算
    for perm in range(n_permutations):
        null_r, _ = pearsonr(profile, surrogate_maps[perm])
        null_rs[perm] = null_r

    # 计算双侧 P 值
    p_surrogate[i] = np.sum(np.abs(null_rs) >= np.abs(emp_r)) / n_permutations

# ==========================================
# 6. 整理并保存结果
# ==========================================
results_df = pd.DataFrame({
    'Region': regions,
    'Epicenter_r': epicenter_r,
    'p_surrogate': p_surrogate,
    'Significant': ['Yes' if p < 0.05 else 'No' for p in p_surrogate]
})

# 按照 R 的绝对值从大到小排序
results_df['Abs_r'] = results_df['Epicenter_r'].abs()
results_df = results_df.sort_values(by='Abs_r', ascending=False).drop(columns=['Abs_r'])

results_df.to_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step16_Epicenters/subtype2_FullBrain_Epicenters_BrainSMASH2.csv', index=False)
print("\n🎉 分析完成！")
