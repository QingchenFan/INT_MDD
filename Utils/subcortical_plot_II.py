import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from nilearn import plotting, datasets, image
import numpy as np
import os
'''
第二步-画出生成图  自定义 colorbar
'''
plt.switch_backend('Agg')
path = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_Match"
# 设置文件路径
nii_file = os.path.join(path, "MDD_GroupMeanINT_subcotical.nii")

# ==========================================
# ⭐ 新增：读取数据并打印实际的 vmin 和 vmax
# ==========================================
img = image.load_img(nii_file)
data = img.get_fdata()

print(f"--- 图像数据统计信息 ---")
print(f"数据整体最小值 (含背景): {np.nanmin(data):.4f}")
print(f"数据整体最大值 (含背景): {np.nanmax(data):.4f}")

# 排除背景 0 值，查看大脑有效区域的极值
non_zero_data = data[data != 0]
if len(non_zero_data) > 0:
    print(f"大脑区域(非零)最小值: {np.nanmin(non_zero_data):.4f}")
    print(f"大脑区域(非零)最大值: {np.nanmax(non_zero_data):.4f}")
print(f"------------------------\n")
# ==========================================

# 定义你的自定义颜色并创建 colormap
hex_colors = ["#00478D", "#7DA2C6", "#FAFDFF","#FAFDFF", "#D78393", "#B50A27"]
#hex_colors = ["#00478D", "#7DA2C6", "#FAFDFF"]  # 蓝色
#hex_colors = [ "#FAFDFF", "#D78393", "#B50A27"] # 红色

my_cmap = LinearSegmentedColormap.from_list("custom_coolwarm", hex_colors)

# 创建画布
fig = plt.figure(figsize=(15, 4), facecolor='white')

# 加载高分辨率 (1mm) 的 MNI152 模板
template = datasets.load_mni152_template(resolution=1)

# 注意：等你看到终端打印出数值后，你可以再把 vmin 和 vmax 解除注释并填入合适的值
plotting.plot_stat_map(
    nii_file,
    bg_img=template,
    display_mode='z',
    cut_coords=[-18, 7],
    symmetric_cbar=False,
    cmap=my_cmap,
    dim=0.8,
    alpha=0.9,
    black_bg=False,
    figure=fig,
    vmin=3.3,               # 根据终端打印的"大脑区域最小值"来设置
    vmax=4.1,                   # 根据终端打印的"大脑区域最大值"来设置
    annotate=False
)

savepath = os.path.join(path, "MDD_subcotical_meanINT.png")
# 保存图片
plt.savefig(savepath, dpi=300, bbox_inches='tight')
plt.close()