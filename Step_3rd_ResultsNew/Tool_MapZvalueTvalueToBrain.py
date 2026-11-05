import pandas as pd
import numpy as np
import nibabel as nib
'''
    将 z值或者 t值 按照p值是否显著映射到大脑上
'''
# ==========================================
# 1. 路径设置 (请根据您的实际环境确认路径)
# ==========================================
# GLM 结果的 CSV 文件路径
csv_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_covariate/Step2_subtype/Step2_BNA246INT_Ttest/三组比较结果/posthoc_HC_minus_subtype1.csv'
# 您的 CIFTI 大脑模板路径
tpath = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii'
# 输出的 dscalar 文件路径 (建议修改名称以作区分)
output_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_covariate/Step2_subtype/Step2_BNA246INT_Ttest/三组比较结果/HCvsS1_tValue.dscalar.nii'

# ==========================================
# 2. 读取并过滤 GLM 结果数据
# ==========================================
df = pd.read_csv(csv_path)

# 提取 z 值和 p 值
z_values = df['tvalue'].values
p_values = df['fdr-pvalue'].values

# 构建过滤后的数值向量 (基于您的要求：显著 p<0.05 且 z<0)
# 满足条件则保留 z_score，否则替换为 0
target_values = np.where((p_values < 0.05) & (z_values < 0), z_values, 0)

# ==========================================
# 3. 加载大脑模板
# ==========================================
template_img = nib.load(tpath)
render_data = template_img.get_fdata().copy()

# 原代码中的标签处理逻辑
render_data[render_data > 210] -= 210

print(f"模板顶点数: {render_data.shape[1]}")
print(f"CSV 中的脑区总数: {len(target_values)}")
print(f"👉 其中满足显著 (p<0.05) 且 Z<0 的脑区数量: {np.sum(target_values != 0)}")

# ==========================================
# 4. 将 Z 值映射到对应的标签索引上
# ==========================================
# 将模板上原来所有的非 0 值先清零，避免残留模板自带的 ID 号
render_data[:] = 0

# 重新加载一遍未清零的原始图来做 mask
orig_data = template_img.get_fdata().copy()
orig_data[orig_data > 210] -= 210

# 遍历 1 到 246 (Brainnetome 246 标准索引)
for i in range(1, len(target_values) + 1):
    mask = (orig_data == i)
    # 将过滤后的 Z 值赋给这些顶点
    render_data[mask] = target_values[i - 1]

# ==========================================
# 5. 构建并保存 CIFTI dscalar 文件
# ==========================================
# 命名此列数据的名称
scalar_axis = nib.cifti2.cifti2_axes.ScalarAxis(['Significant_Negative_Z'])
brain_model_axis = template_img.header.get_axis(1)
new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_model_axis))
new_img = nib.Cifti2Image(render_data, header=new_header)
new_img.to_filename(output_path)

print(f"✅ 映射完成！文件已保存至: {output_path}")