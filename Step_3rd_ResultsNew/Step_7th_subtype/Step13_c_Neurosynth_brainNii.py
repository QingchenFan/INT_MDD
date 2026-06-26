import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.spatial.distance import cdist
from brainspace.null_models import MoranRandomization
import multiprocessing
from functools import partial


# ==========================================
# 1. 函数：从 NIfTI 模板提取脑区质心 (Centroids)
# ==========================================
def extract_centroids(atlas_nii_path, n_regions=246):
    print("正在提取 BNA246 图谱脑区质心...")
    atlas_img = nib.load(atlas_nii_path)
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine

    centroids = np.zeros((n_regions, 3))
    for label_idx in range(1, n_regions + 1):
        coords = np.argwhere(atlas_data == label_idx)
        if len(coords) > 0:
            mean_coord = coords.mean(axis=0)
            phys_coord = nib.affines.apply_affine(affine, mean_coord)
            centroids[label_idx - 1] = phys_coord
        else:
            print(f"警告: Label {label_idx} 在模板中未找到！")
    return centroids


# ==========================================
# 2. 核心函数：处理单个打乱后的假向量，生成 .nii
# ==========================================
def save_single_surrogate(i, surrogate_z_matrix, atlas_data, affine, header, output_dir, direction, z_threshold):
    """
    修改后：正确使用 z_threshold 对假数据进行空间过滤，允许假团块在全脑随机游走。
    """
    # 提取第 i 次打乱生成的假 Z 值向量 (长度 246)
    fake_z_vector = surrogate_z_matrix[i, :]

    # 初始化全是 0 的过滤后数组
    fake_filtered = np.zeros_like(fake_z_vector)

    for idx in range(246):
        z_val = fake_z_vector[idx]

        # 【核心修正逻辑】：不看真实 p 值，只看假数据自己有没有随机出足够大的 Z 值！
        if direction == 'pos' and z_val > z_threshold:
            fake_filtered[idx] = z_val
        elif direction == 'neg' and z_val < -z_threshold:
            fake_filtered[idx] = z_val

    # 将过滤好的 246 个假数值填回 3D 脑图
    full_data = np.zeros(atlas_data.shape, dtype=np.float32)
    for label_idx in range(1, 247):
        full_data[atlas_data == label_idx] = fake_filtered[label_idx - 1]

    # 生成并保存 .nii
    fake_img = nib.Nifti1Image(full_data, affine, header)

    # 根据 direction 命名
    file_id = str(i + 1).zfill(5)
    file_name = f'rsurr_g1_z_{direction}_{file_id}.nii'
    save_path = os.path.join(output_dir, file_name)
    nib.save(fake_img, save_path)

    if (i + 1) % 1000 == 0:
        print(f"   已生成 {i + 1} 个假图...")


# ==========================================
# 主执行入口
# ==========================================
if __name__ == '__main__':
    # ----------------------------------------
    # 【用户配置区】
    # ----------------------------------------
    TARGET_DIRECTION = 'pos'  # 选择 'pos' 或 'neg'
    N_PERMUTATIONS = 10000  # 正式跑请改回 10000
    FDR_THRESHOLD = 0.05
    N_CORES = 16

    # 路径配置
    value_csv = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step15_Neurosynth/Neuronsynth_II/S1_vs_HC_zmap_INT.csv'
    label_csv = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/region246_network_Yeo.csv'
    template_nii = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_246_2mm.nii.gz'

    # 建立输出文件夹
    output_dir = f'/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step15_Neurosynth/Neuronsynth_II/BrainNii100000/surr_{TARGET_DIRECTION}'
    os.makedirs(output_dir, exist_ok=True)
    # ----------------------------------------

    print(f"--- 启动 Moran 空间零模型生成流水线 [{TARGET_DIRECTION.upper()}] ---")

    # 1. 准备真实的 Z 值，并自动寻找阈值
    print("1. 正在对齐真实 Z 值并计算统计阈值...")
    df_values = pd.read_csv(value_csv)
    df_labels = pd.read_csv(label_csv)

    real_z_vector = np.zeros(246)

    for index, row in df_labels.iterrows():
        label_id = int(row['Label'])
        region_name = str(row['regions'])
        match = df_values[df_values['region'] == region_name]
        if not match.empty:
            real_z_vector[label_id - 1] = match['z'].values[0]
        else:
            print(f"警告: 脑区 {region_name} 在数值表里找不到！")

    # 【新增功能：自动计算 Z 值边界 (Z-threshold)】
    # 找到所有 FDR_p < 0.05 的脑区，取它们 Z 绝对值的最小值作为“及格线”
    significant_z_values = df_values[df_values['FDR_p'] < FDR_THRESHOLD]['z'].abs()

    if len(significant_z_values) > 0:
        auto_z_threshold = significant_z_values.min()
        print(f"   --> 成功提取 Z 值阈值: {auto_z_threshold:.4f} (对应 FDR_p < {FDR_THRESHOLD})")
    else:
        # 如果真实数据中一个显著的脑区都没有，默认用 Z=1.96 (p<0.05)
        auto_z_threshold = 1.96
        print(f"   --> 警告：真实数据中没有显著脑区，默认使用 Z=1.96 阈值")

    # 2. 计算脑区之间的物理距离矩阵
    print("2. 正在计算 BNA246 图谱的空间距离矩阵...")
    centroids = extract_centroids(template_nii, n_regions=246)
    distance_matrix = cdist(centroids, centroids, metric='euclidean')

    # 3. 运行 Moran Spectral Randomization 生成假数据
    print(f"3. 核心步骤: 正在利用 Moran 算法生成 {N_PERMUTATIONS} 个保留空间平滑度的假 Z 值矩阵...")
    moran = MoranRandomization(n_rep=N_PERMUTATIONS, procedure='singleton', tol=1e-6)

    # 拟合模型并生成打乱数据
    surrogate_z_matrix = moran.fit(distance_matrix).randomize(real_z_vector)

    print("   Moran 随机化完成！")

    # 4. 读取模板头文件准备批量写出
    print("4. 正在应用 Z 值阈值，并通过多进程批量生成 NIfTI 文件...")
    atlas_img = nib.load(template_nii)
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine
    header = atlas_img.header
    header.set_data_dtype(np.float32)

    # 5. 使用多进程加速 NIfTI 的生成
    # 删除了 original_fdr_p，传入自动计算出的 auto_z_threshold
    worker_func = partial(save_single_surrogate,
                          surrogate_z_matrix=surrogate_z_matrix,
                          atlas_data=atlas_data,
                          affine=affine,
                          header=header,
                          output_dir=output_dir,
                          direction=TARGET_DIRECTION,
                          z_threshold=auto_z_threshold)

    items = list(range(N_PERMUTATIONS))

    with multiprocessing.Pool(N_CORES) as p:
        p.map(worker_func, items)

    print(f"\n--- 大功告成！{N_PERMUTATIONS} 个假文件已妥善保存在 {output_dir} 文件夹中。 ---")
    print(f"现在，您可以运行 Neurosynth 多进程解码代码了！")