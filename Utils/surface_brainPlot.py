import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from nilearn import plotting
'''
    使用 Nilearn 将 .dscalar.nii 映射到大脑皮层进行画图展示
    好处是可以自定义 colorbar（但是只能定义首尾数值对应 colorbar）
'''
# 后端设置，无GUI服务器环境可用
plt.switch_backend('Agg')

# ============================================================
# 1. 文件路径
# ============================================================

dscalar_file = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_Match/Step2_subtype/Step7_Neurosynth/posneg/subtype2_zmap_negative.dscalar.nii"

# fsLR 32k surface 文件
template_dir = "/Users/qingchen/Documents/Data/template/HCP_S1200_Atlas_Z4_pkXDZ"
left_surf = os.path.join(template_dir, "S1200.L.pial_MSMAll.32k_fs_LR.surf.gii")
right_surf = os.path.join(template_dir, "S1200.R.pial_MSMAll.32k_fs_LR.surf.gii")
atlas_parcellation_file = "/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii"

# 背景阴影文件 (如果需要更立体的效果，可以加上，如果没有可以设为 None)
left_sulc = None
right_sulc = None

# ============================================================
# 2. 输出文件夹
# ============================================================
output_dir = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_Match/Step2_subtype/Step7_Neurosynth/posneg/zmap_Brain"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 3. 读取 dscalar.nii (数据)
# ============================================================
img = nib.load(dscalar_file)
data = img.get_fdata()

# 如果只有一个 scalar map
if data.shape[0] == 1:
    data = data[0]

# ============================================================
# 4. 获取 CIFTI BrainModelAxis 并恢复数据到 32k
# ============================================================
brain_model_axis = img.header.get_axis(1)

# fsLR 32k 模板每个半球有 32492 个顶点
N_VERTICES_PER_HEMI = 32492
left_full_data = np.full(N_VERTICES_PER_HEMI, np.nan)
right_full_data = np.full(N_VERTICES_PER_HEMI, np.nan)

for structure, slc, brain_model in brain_model_axis.iter_structures():
    structure_data = data[slc]
    if structure == "CIFTI_STRUCTURE_CORTEX_LEFT":
        left_full_data[brain_model.vertex] = structure_data
    elif structure == "CIFTI_STRUCTURE_CORTEX_RIGHT":
        right_full_data[brain_model.vertex] = structure_data

# ============================================================
# 4.5. 读取图谱标签文件并恢复到 32k (新增)
# ============================================================
# 读取 atlas dlabel 文件
atlas_img = nib.load(atlas_parcellation_file)
atlas_data = atlas_img.get_fdata()[0]
atlas_brain_model_axis = atlas_img.header.get_axis(1)

left_atlas_labels = np.zeros(N_VERTICES_PER_HEMI)
right_atlas_labels = np.zeros(N_VERTICES_PER_HEMI)

for structure, slc, brain_model in atlas_brain_model_axis.iter_structures():
    structure_labels = atlas_data[slc]
    if structure == "CIFTI_STRUCTURE_CORTEX_LEFT":
        left_atlas_labels[brain_model.vertex] = structure_labels
    elif structure == "CIFTI_STRUCTURE_CORTEX_RIGHT":
        right_atlas_labels[brain_model.vertex] = structure_labels

# ============================================================
# 5. 将 0 设置为 NaN (如果 0 代表背景或内侧壁)
# ============================================================
left_full_data[left_full_data == 0] = np.nan
right_full_data[right_full_data == 0] = np.nan

# ============================================================
# 6. 计算统一颜色范围
# ============================================================
all_values = np.concatenate([
    left_full_data[np.isfinite(left_full_data)],
    right_full_data[np.isfinite(right_full_data)]
])

vmin = np.min(all_values)
vmax = np.max(all_values)
print(f"Data range: vmin = {vmin: .4f}, vmax = {vmax: .4f}")
# vmin = 3.0
# vmax = 4.2

# ============================================================
# 7. 绘图与保存 (使用 Nilearn)
# ============================================================
views = {
    'Lateral': 'lateral',
    'Medial': 'medial'
}

# 注意：这里将图谱的 numpy 数组 (left_atlas_labels / right_atlas_labels) 加进了字典
hemispheres = {
    'Left': (left_surf, left_full_data, left_sulc, 'left', left_atlas_labels),
    'Right': (right_surf, right_full_data, right_sulc, 'right', right_atlas_labels)
}

#my_colors = ["#024892", "#6AB4DC", "#FAFDFF", "#42B8B1", "#02927A"]  # HC MDD 配色
my_colors = ["#00478D","#7DA2C6", "#FAFDFF", "#D78393", "#B50A27"]  # 亚型配色 "#B50A27"
my_colors = ["#00478D", "#1E5F9F", "#4B82B3", "#7DA2C6", "#B8CEDD", "#FAFDFF", "#E5AAB5", "#D78393", "#C54A61", "#B50A27"
]
my_custom_cmap = LinearSegmentedColormap.from_list("Positive_Mean_Cmap", my_colors)

# 解包时增加了 hemi_atlas_labels
for hemi_name, (surf_mesh, surf_map, bg_map, hemi_dir, hemi_atlas_labels) in hemispheres.items():
    if bg_map is None:
        bg_map = np.ones(surf_map.shape) * 0.1  # 生成的灰色背景

    for view_name, view_ang in views.items():
        output_file = os.path.join(output_dir, f"{hemi_name}_{view_name}_tvalue_s2.png")

        # 创建画布
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': '3d'})

        # 绘制皮层数据
        plotting.plot_surf(
            surf_mesh=surf_mesh,
            surf_map=surf_map,
            hemi=hemi_dir,
            view=view_ang,
            bg_map=bg_map,
            cmap=my_custom_cmap,
            #cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            cbar_tick_format="%.2f",
            colorbar=False,
            title=f"{hemi_name} Hemisphere - {view_name}",
            axes=ax
        )

        # 脑区边界加黑色线
        # roi_levels = np.unique(hemi_atlas_labels)
        # roi_levels = roi_levels[roi_levels != 0]
        # plotting.plot_surf_contours(
        #     surf_mesh=surf_mesh,
        #     roi_map=hemi_atlas_labels,
        #     axes=ax,
        #     levels=roi_levels,  # 明确指定需要画边界的脑区
        #     colors=['black'] * len(roi_levels),
        #     linewidths=0.1
        # )


        plt.savefig(output_file, dpi=300, bbox_inches="tight", transparent=True)
        print("Saved:", output_file)
        plt.close(fig)

# ============================================================
# 8. 绘制统一的 Colorbar
# ============================================================
fig = plt.figure(figsize=(8, 1.5))
ax = fig.add_axes([0.1, 0.4, 0.8, 0.25])

norm = Normalize(vmin=vmin, vmax=vmax)

sm = plt.cm.ScalarMappable(cmap=my_custom_cmap,
                           #cmap="coolwarm",
                           norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=ax, orientation="horizontal")
#cbar.set_label("HC Regional Mean Value", fontsize=14)

colorbar_file = os.path.join(output_dir, "Colorbar.png")
plt.savefig(colorbar_file, dpi=300, bbox_inches="tight", transparent=True)
print("Saved:", colorbar_file)
plt.close(fig)