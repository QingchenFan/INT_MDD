import pandas as pd
import numpy as np
import nibabel as nib

# 1. 路径设置
# 之前步骤生成的包含均值的 CSV 文件
csv_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype2_brain_regions_mean_INT.csv'
# 您的大脑模板路径
tpath = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii'
# 输出的 dscalar 文件路径
output_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype2_meanINT.dscalar.nii'

# 2. 读取 CSV 数据
df = pd.read_csv(csv_path)
# 排除 'region' 辅助列，获取 246 个脑区的均值 (shape: 1, 246)
# 如果您的 CSV 第一列是 subID 或 region，请确保将其 drop 掉
data_values = df.drop(columns=['region']).values

# 3. 加载大脑模板
template_img = nib.load(tpath)
# 获取标签数据 (通常 shape 为 (1, 64984))
# 使用 .copy() 避免修改原始加载的对象内存
render_data = template_img.get_fdata().copy()

# 原代码中的标签处理逻辑 (如果您的模板 L/R 标签是分开且需要合并的，请取消注释)
render_data[render_data > 210] -= 210

print(f"模板顶点数: {render_data.shape[1]}")
print(f"待映射的脑区数: {data_values.shape[1]}")

# 4. 将均值映射到对应的标签索引上
# 遍历 1 到 246 (Brainnetome 246 标准索引)
for i in range(1, data_values.shape[1] + 1):
    # 找到模板中所有属于该脑区 (label == i) 的顶点
    mask = (render_data == i)
    # 将对应的均值赋给这些顶点
    # data_values[0, i-1] 对应 CSV 中的第 i 个脑区数据
    render_data[mask] = data_values[0, i - 1]

# 5. 构建并保存 CIFTI dscalar 文件
# 创建 Scalar 轴 (定义列名)
scalar_axis = nib.cifti2.cifti2_axes.ScalarAxis(['Mean_INT'])
brain_model_axis = template_img.header.get_axis(1)
new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_model_axis))
new_img = nib.Cifti2Image(render_data, header=new_header)
new_img.to_filename(output_path)

print(f"映射完成！文件已保存至: {output_path}")