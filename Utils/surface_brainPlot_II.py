import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from nilearn import plotting

# ===================== 配置 =====================
# 文件路径
dscalar_file = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_covariate/Step2_subtype/Step18_Genetic/result_Gene2_Correlation/subtype2/subtype2_PLSscore.dscalar.nii"
template_dir = "/Users/qingchen/Documents/Data/template/HCP_S1200_Atlas_Z4_pkXDZ"
left_surf = os.path.join(template_dir, "S1200.L.pial_MSMAll.32k_fs_LR.surf.gii")
right_surf = os.path.join(template_dir, "S1200.R.pial_MSMAll.32k_fs_LR.surf.gii")
atlas_parcellation_file = "/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii"

# 输出目录
output_dir = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_covariate/Step2_subtype/Step18_Genetic/result_Gene2_Correlation/subtype2/s2_PLSscoreMapBrain"
os.makedirs(output_dir, exist_ok=True)

# 绘图参数
VMIN, VMAX = -1.0, 1.0
DRAW_ROI_CONTOURS = False
BRAIN_FIGSIZE, BRAIN_DPI = (8, 8), 300
CBAR_FIGSIZE, CBAR_DPI, CBAR_FONT_SIZE = (10, 1.5), 300, 20
COLOR_MAPPING_VALUES = [(-1.0, "#00478D"), (-0.5, "#7DA2C6"), (0.0, "#FAFDFF"), (0.5, "#D78393"), (1.0, "#B50A27")]
ZERO_AS_BACKGROUND = True
N_VERTICES_PER_HEMI = 32492

# ===================== 读取CIFTI数据 =====================
img = nib.load(dscalar_file)
data = np.asarray(img.get_fdata())
if data.ndim == 2:
    if data.shape[0] != 1:
        raise ValueError("当前 dscalar 文件包含多个 scalar maps，请明确指定需要绘制的 map。")
    data = data[0]
data = data.ravel()
brain_model_axis = img.header.get_axis(1)

# 分配到左右半球顶点
left_full_data = np.full(N_VERTICES_PER_HEMI, np.nan, dtype=float)
right_full_data = np.full(N_VERTICES_PER_HEMI, np.nan, dtype=float)
for structure, slc, brain_model in brain_model_axis.iter_structures():
    structure_data = data[slc]
    if structure == "CIFTI_STRUCTURE_CORTEX_LEFT":
        left_full_data[brain_model.vertex] = structure_data
    elif structure == "CIFTI_STRUCTURE_CORTEX_RIGHT":
        right_full_data[brain_model.vertex] = structure_data

# ===================== 读取脑图谱标签 =====================
left_atlas_labels = np.zeros(N_VERTICES_PER_HEMI, dtype=int)
right_atlas_labels = np.zeros(N_VERTICES_PER_HEMI, dtype=int)
if DRAW_ROI_CONTOURS:
    atlas_img = nib.load(atlas_parcellation_file)
    atlas_data = np.asarray(atlas_img.get_fdata()[0])
    atlas_brain_model_axis = atlas_img.header.get_axis(1)
    for structure, slc, brain_model in atlas_brain_model_axis.iter_structures():
        structure_labels = atlas_data[slc].astype(int)
        if structure == "CIFTI_STRUCTURE_CORTEX_LEFT":
            left_atlas_labels[brain_model.vertex] = structure_labels
        elif structure == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            right_atlas_labels[brain_model.vertex] = structure_labels

# ===================== 背景处理 & 数据范围检查 =====================
if ZERO_AS_BACKGROUND:
    left_full_data[left_full_data == 0] = np.nan
    right_full_data[right_full_data == 0] = np.nan

all_values = np.concatenate([left_full_data[np.isfinite(left_full_data)], right_full_data[np.isfinite(right_full_data)]])
if all_values.size == 0:
    raise ValueError("没有找到有效的皮层数据。")
actual_min, actual_max = np.min(all_values), np.max(all_values)
print("=" * 60)
print(f"Actual data range:\nMinimum: {actual_min:.4f}\nMaximum: {actual_max:.4f}\n")
print(f"Colorbar range:\nVMIN: {VMIN:.4f}\nVMAX: {VMAX:.4f}")
print("=" * 60)
if actual_min < VMIN or actual_max > VMAX:
    print("Warning: Some values exceed the fixed color range and will be displayed using endpoint colors.")

# ===================== 自定义colormap =====================
normalized_mapping = []
for val, color in COLOR_MAPPING_VALUES:
    pos = (val - VMIN) / (VMAX - VMIN)
    normalized_mapping.append((pos, color))
my_custom_cmap = LinearSegmentedColormap.from_list("Custom_Mapped_Cmap", normalized_mapping, N=256)
my_custom_cmap.set_bad(color=(0, 0, 0, 0))
norm = Normalize(vmin=VMIN, vmax=VMAX, clip=True)

# ===================== 绘制皮层表面图 =====================
views = {"Lateral": "lateral", "Medial": "medial"}
hemispheres = {
    "Left": (left_surf, left_full_data, "left", left_atlas_labels),
    "Right": (right_surf, right_full_data, "right", right_atlas_labels)
}

for hemi_name, (surf_mesh, surf_map, hemi_dir, hemi_atlas_labels) in hemispheres.items():
    bg_map = np.ones(surf_map.shape, dtype=float) * 0.1
    for view_name, view_ang in views.items():
        out_path = os.path.join(output_dir, f"{hemi_name}_{view_name}_PLSscore_s2.png")
        fig, ax = plt.subplots(figsize=BRAIN_FIGSIZE, subplot_kw={"projection": "3d"})
        plotting.plot_surf(
            surf_mesh=surf_mesh, surf_map=surf_map, hemi=hemi_dir, view=view_ang,
            bg_map=bg_map, cmap=my_custom_cmap, vmin=VMIN, vmax=VMAX,
            colorbar=False, title=f"{hemi_name} Hemisphere - {view_name}", axes=ax
        )
        if DRAW_ROI_CONTOURS:
            roi_levels = np.unique(hemi_atlas_labels)
            roi_levels = roi_levels[roi_levels != 0]
            plotting.plot_surf_contours(
                surf_mesh=surf_mesh, roi_map=hemi_atlas_labels, axes=ax,
                levels=roi_levels, colors=["black"] * len(roi_levels), linewidths=0.1
            )
        plt.savefig(out_path, dpi=BRAIN_DPI, bbox_inches="tight", transparent=True)
        print("Saved:", out_path)
        plt.close(fig)

# ===================== 单独绘制colorbar =====================
fig = plt.figure(figsize=CBAR_FIGSIZE)
ax = fig.add_axes([0.08, 0.45, 0.84, 0.25])
sm = plt.cm.ScalarMappable(cmap=my_custom_cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=ax, orientation="horizontal")
cbar_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels(["-1.0", "-0.5", "0", "0.5", "1.0"])
cbar.ax.tick_params(axis="x", labelsize=CBAR_FONT_SIZE, width=2, length=6, direction="out", pad=8)
for label in cbar.ax.get_xticklabels():
    label.set_fontweight("bold")
    label.set_fontfamily("Arial")
cbar.outline.set_linewidth(2)

cbar_path = os.path.join(output_dir, "Colorbar.png")
plt.savefig(cbar_path, dpi=CBAR_DPI, bbox_inches="tight", transparent=True)
print("Saved:", cbar_path)
plt.close(fig)

print("\nAll figures generated successfully!")
