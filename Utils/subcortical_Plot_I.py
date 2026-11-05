import pandas as pd
import nibabel as nib
import numpy as np
import os
'''
    第一步：此代码得到 subcortical 的 Volume 数据（.nii）之后可用于画图（subcortical_nilearn.py画图）。无卡阈值！！！
'''

def process_brain_data(value_csv, label_csv, template_nii, output_full_nii, output_subcortical_nii):
    """
    整合处理函数：
    1. 读取 CSV 数据并匹配到 Brainnetome Atlas。
    2. 生成全脑 NIfTI 文件。
    3. 基于全脑数据，剔除皮层区域（Label < 211），生成仅含皮层下的 NIfTI 文件。
    """
    print(f"--- 开始处理流水线 ---")

    # ==========================================
    # 第一步：读取数据与建立映射
    # ==========================================
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
            id_to_value_map[label_id] = target_val   # 构建字段，label:mean value
            print(id_to_value_map)

        else:
            missing_regions.append(region_name)

    print(f"   匹配完成: 成功 {len(id_to_value_map)} 个, 失败 {len(missing_regions)} 个。")

    # ==========================================
    # 第二步：生成全脑数据矩阵
    # ==========================================
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

    # ==========================================
    # 第三步：生成仅皮层下数据 (Subcortical Only)
    # ==========================================
    print("3. [皮层下] 正在剔除皮层数据 (保留 Label 211-246)...")

    # 直接复制刚才生成的全脑数据，无需重新加载
    subcortical_data = full_data.copy()

    # 核心逻辑：找到 Label < 211 且不为背景(0) 的区域 -> 设为 0
    mask_cortex = (atlas_data < 211) & (atlas_data > 0)
    subcortical_data[mask_cortex] = 0

    # 确保背景也是 0
    subcortical_data[atlas_data == 0] = 0

    # 保存皮层下 NIfTI 文件
    subc_img = nib.Nifti1Image(subcortical_data, atlas_img.affine, atlas_img.header)
    subc_img.header.set_data_dtype(np.float32)

    print(f"   正在保存仅含皮层下的结果到: {output_subcortical_nii}")
    nib.save(subc_img, output_subcortical_nii)

    print("--- 全部处理完成 ---")


# ==========================================
# 在这里配置你的文件路径
# ==========================================

# 输入文件

input_label_csv = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/region246_network_Yeo.csv'
input_template = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_246_2mm.nii.gz'


path = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_Match/"
#修改一下路径
input_value_csv = os.path.join(path, "MDD_meanINT.csv")
# 输出文件
output_full_file = os.path.join(path, "MDD_GroupMeanINT.nii")
output_subc_file = os.path.join(path, "MDD_GroupMeanINT_subcotical.nii")

if __name__ == '__main__':

        process_brain_data(
            input_value_csv,
            input_label_csv,
            input_template,
            output_full_file,
            output_subc_file
        )
