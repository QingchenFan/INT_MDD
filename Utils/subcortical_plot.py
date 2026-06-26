import os
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting, datasets
from nilearn.conftest import matplotlib

matplotlib.use('Agg')
# ==========================================
# 1. 数据处理函数：生成全脑与皮层下 NIfTI 文件
# ==========================================
def process_brain_data(value_csv, label_csv, template_nii, output_full_nii, output_subcortical_nii):
    """
    整合处理函数：
    1. 读取 CSV 数据并匹配到 Brainnetome Atlas。
    2. 生成全脑 NIfTI 文件。
    3. 基于全脑数据，剔除皮层区域（Label < 211），生成仅含皮层下的 NIfTI 文件。
    """
    print(f"--- 开始处理流水线 ---")

    # 第一步：读取数据与建立映射
    print("1. [读取] 正在读取数据文件...")
    df_values = pd.read_csv(value_csv)
    df_labels = pd.read_csv(label_csv)

    # 获取指标值（取第一行）
    mean_values = df_values.iloc[0]
    data_columns = df_values.columns.tolist()

    # 建立 [Label ID] -> [指标值] 的字典
    id_to_value_map = {}
    missing_regions = []

    print("   正在匹配脑区名称与 Label ID...")
    for index, row in df_labels.iterrows():
        label_id = row['Label']
        region_name = str(row['regions'])

        target_val = None
        # 匹配逻辑
        if region_name in data_columns:
            target_val = mean_values[region_name]
        else:
            if '/' in region_name:
                short_name = region_name.split('/')[-1]
                if short_name in data_columns:
                    target_val = mean_values[short_name]

        if target_val is not None:
            id_to_value_map[label_id] = target_val
        else:
            missing_regions.append(region_name)

    print(f"   匹配完成: 成功 {len(id_to_value_map)} 个, 失败 {len(missing_regions)} 个。")

    # 第二步：生成全脑数据矩阵
    print(f"2. [全脑] 正在读取模板并填入数据: {template_nii}")
    atlas_img = nib.load(template_nii)
    atlas_data = atlas_img.get_fdata()

    # 创建全脑数据数组 (float32)
    full_data = np.zeros(atlas_data.shape, dtype=np.float32)

    # 填入数据
    for label_id, val in id_to_value_map.items():
        full_data[atlas_data == label_id] = val

    # 保存全脑 NIfTI 文件
    full_img = nib.Nifti1Image(full_data, atlas_img.affine, atlas_img.header)
    full_img.header.set_data_dtype(np.float32)

    print(f"   正在保存全脑结果到: {output_full_nii}")
    nib.save(full_img, output_full_nii)

    # 第三步：生成仅皮层下数据 (Subcortical Only)
    print("3. [皮层下] 正在剔除皮层数据 (保留 Label 211-246)...")
    subcortical_data = full_data.copy()

    # 核心逻辑：找到 Label < 211 且不为背景(0) 的区域 -> 设为 0
    mask_cortex = (atlas_data < 211) & (atlas_data > 0)
    subcortical_data[mask_cortex] = 0
    subcortical_data[atlas_data == 0] = 0  # 确保背景也是 0

    # 保存皮层下 NIfTI 文件
    subc_img = nib.Nifti1Image(subcortical_data, atlas_img.affine, atlas_img.header)
    subc_img.header.set_data_dtype(np.float32)

    print(f"   正在保存仅含皮层下的结果到: {output_subcortical_nii}")
    nib.save(subc_img, output_subcortical_nii)
    print("--- NIfTI 文件生成完成 ---\n")


# ==========================================
# 2. 画图函数：可视化皮层下结果并保存
# ==========================================
def plot_subcortical_data(nii_file, save_path):
    print(f"--- 开始绘制可视化图像 ---")
    print(f"   正在加载 MNI152 模板并绘制 {nii_file}...")

    # 创建画布
    fig = plt.figure(figsize=(15, 4), facecolor='white')

    # 加载高分辨率 (1mm) 的 MNI152 模板
    template = datasets.load_mni152_template(resolution=1)

    plotting.plot_stat_map(
        nii_file,
        bg_img=template,
        display_mode='z',
        cut_coords=[-18, 7],
        symmetric_cbar=True,
        cmap='coolwarm',
        dim=0.8,  # 调整背景亮度 (-1 到 1)，负值增加对比度
        alpha=0.9,  # 让颜色稍微透明，露出背景结构
        black_bg=False,
        figure=fig,
        # vmin=-0.7,
        # vmax=1,
        annotate=False
    )

    # 保存图片
    print(f"   正在保存图像至: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print("--- 可视化完成 ---")


# ==========================================
# 主执行入口
# ==========================================
if __name__ == '__main__':
    # 配置文件路径
    input_label_csv = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/region246_network_Yeo.csv'
    input_template = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_246_2mm.nii.gz'
    path = "/Volumes/QCI/NormativeModel/Results/Result_GrayVol246_BLR_HCMDD_250512/StaResults/subtype/subtype_RegionTtest/"

    input_value_csv = os.path.join(path, "subtypeDiff_Region_ttest_results_fdr_pvalue.csv")
    output_full_file = os.path.join(path, "subtypeDiff_Region_ttest_results.nii")
    output_subc_file = os.path.join(path, "subtypeDiff_Region_ttest_results_subcotical_mean.nii")
    output_png_file = os.path.join(path, "subtypeDiff_Region_ttest_results_subcotical_mean.png")

    # 执行流水线：1. 处理数据生成 nii
    process_brain_data(
        input_value_csv,
        input_label_csv,
        input_template,
        output_full_file,
        output_subc_file
    )

    # 执行流水线：2. 绘制刚生成的皮层下 nii 文件
    plot_subcortical_data(
        output_subc_file,
        output_png_file
    )