import os
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting, datasets
import matplotlib

matplotlib.use('Agg')


# ==========================================
# 1. 数据处理函数：生成全脑与皮层下 NIfTI 文件
# ==========================================
def process_brain_data(value_csv, label_csv, template_nii, output_full_nii, output_subcortical_nii, fdr_threshold=0.05,
                       direction='pos'):
    """
    整合处理函数：支持通过 direction 参数切换正向/负向提取
    - direction='pos': 提取 FDR_p < threshold 且 Z > 0 的区域
    - direction='neg': 提取 FDR_p < threshold 且 Z < 0 的区域
    """
    print(f"\n--- 开始处理流水线 [模式: {direction.upper()}] ---")

    print("1. [读取] 正在读取数据文件...")
    df_values = pd.read_csv(value_csv)
    df_labels = pd.read_csv(label_csv)

    id_to_value_map = {}
    missing_regions = []

    print(
        f"   正在应用严格阈值: FDR_p < {fdr_threshold} 且提取 {'正值(Z>0)' if direction == 'pos' else '负值(Z<0)'}...")

    for index, row in df_labels.iterrows():
        label_id = row['Label']
        region_name = str(row['regions'])

        match = df_values[df_values['region'] == region_name]

        if not match.empty:
            z_val = match['z'].values[0]
            fdr_p = match['FDR_p'].values[0]

            # 【核心合并逻辑】：根据传入的 direction 参数动态判断
            is_significant = (fdr_p < fdr_threshold)

            if direction == 'pos' and is_significant and z_val > 0:
                id_to_value_map[label_id] = z_val
            elif direction == 'neg' and is_significant and z_val < 0:
                id_to_value_map[label_id] = z_val
            else:
                id_to_value_map[label_id] = 0.0
        else:
            missing_regions.append(region_name)

    print(f"   匹配完成: 成功处理 {len(id_to_value_map)} 个脑区, 找不到 {len(missing_regions)} 个。")

    print(f"2. [全脑] 正在生成全脑数据...")
    atlas_img = nib.load(template_nii)
    atlas_data = atlas_img.get_fdata()
    full_data = np.zeros(atlas_data.shape, dtype=np.float32)

    for label_id, val in id_to_value_map.items():
        full_data[atlas_data == label_id] = val

    full_img = nib.Nifti1Image(full_data, atlas_img.affine, atlas_img.header)
    full_img.header.set_data_dtype(np.float32)
    nib.save(full_img, output_full_nii)

    print("3. [皮层下] 正在剔除皮层数据...")
    subcortical_data = full_data.copy()
    mask_cortex = (atlas_data < 211) & (atlas_data > 0)
    subcortical_data[mask_cortex] = 0
    subcortical_data[atlas_data == 0] = 0

    subc_img = nib.Nifti1Image(subcortical_data, atlas_img.affine, atlas_img.header)
    subc_img.header.set_data_dtype(np.float32)
    nib.save(subc_img, output_subcortical_nii)
    print("--- NIfTI 文件生成完成 ---\n")


# ==========================================
# 2. 画图函数：可视化皮层下结果并保存
# ==========================================
def plot_subcortical_data(nii_file, save_path, direction='pos'):
    print(f"--- 开始绘制可视化图像 [模式: {direction.upper()}] ---")
    fig = plt.figure(figsize=(15, 4), facecolor='white')
    template = datasets.load_mni152_template(resolution=1)

    img_data = nib.load(nii_file).get_fdata()

    # 【动态画图配置】：根据 direction 设置颜色条和数据范围
    if direction == 'pos':
        max_val = np.max(img_data)
        if max_val <= 0:
            print("--- 警告：数据中没有找到显著的正值，跳过画图 ---")
            return
        plot_kwargs = {'cmap': 'OrRd', 'vmin': 0, 'vmax': max_val}

    elif direction == 'neg':
        min_val = np.min(img_data)
        if min_val >= 0:
            print("--- 警告：数据中没有找到显著的负值，跳过画图 ---")
            return
        plot_kwargs = {'cmap': 'Blues_r', 'vmin': min_val, 'vmax': 0}

    # 执行绘图
    plotting.plot_stat_map(
        nii_file,
        bg_img=template,
        display_mode='z',
        cut_coords=[-18, 7],
        symmetric_cbar=False,
        dim=0.8,
        alpha=0.9,
        black_bg=False,
        figure=fig,
        annotate=False,
        **plot_kwargs  # 将上面判断好的参数直接解包传进来
    )
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"--- 可视化完成，图像已保存至: {save_path} ---")


# ==========================================
# 主执行入口
# ==========================================
if __name__ == '__main__':
    # ----------------------------------------
    # 【用户配置区】
    # ----------------------------------------
    # 只需在这里修改 'pos' 或 'neg' 即可切换整个脚本的行为！
    TARGET_DIRECTION = 'neg'

    # 路径配置
    input_label_csv = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/region246_network_Yeo.csv'
    input_template = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_246_2mm.nii.gz'
    input_value_csv = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step15_Neurosynth/Neuronsynth_II/S2_vs_HC_zmap_INT.csv'

    # 【动态命名】：使用 f-string 根据 direction 自动生成后缀
    output_full_file = f'INT_zmap_cluster_{TARGET_DIRECTION}.nii'
    output_subc_file = f'INT_zmap_cluster_subcortical_{TARGET_DIRECTION}.nii'
    output_png_file = f'INT_zmap_cluster_subcortical_{TARGET_DIRECTION}.png'
    # ----------------------------------------

    # 1. 运行数据处理
    process_brain_data(
        value_csv=input_value_csv,
        label_csv=input_label_csv,
        template_nii=input_template,
        output_full_nii=output_full_file,
        output_subcortical_nii=output_subc_file,
        fdr_threshold=0.05,
        direction=TARGET_DIRECTION  # 传入参数
    )

    # 2. 运行画图
    plot_subcortical_data(
        nii_file=output_subc_file,
        save_path=output_png_file,
        direction=TARGET_DIRECTION  # 传入参数
    )