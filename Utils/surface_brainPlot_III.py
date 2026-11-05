import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from nilearn import plotting

'''
============================================================
使用 Nilearn 将多个 .dscalar.nii 映射到大脑皮层
功能：
1. 多个 dscalar 使用统一颜色范围 
2. 使用【对称式】非线性gamma数值变换增强颜色变化，保证中心白色不偏移
3. Subtype 1 / Subtype 2 使用完全一致的颜色映射，绝对数值可跨图对比
4. 输出一张独立公共 Colorbar，所有脑图共用这同一个colorbar
5. nilearn plot_surf不支持norm对象；数据预先变换到0~1绘图
============================================================
'''
# ============================================================
# 0. Matplotlib 后端
# ============================================================
plt.switch_backend('Agg')

# ============================================================
# 1. 文件路径
# ============================================================
dscalar_files = [
    # Subtype 1
    "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_covariate/Step2_subtype/Step18_Genetic/result_Gene2_Correlation/subtype1/subtype1_PLSscore.dscalar.nii",
    # Subtype 2
    "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_covariate/Step2_subtype/Step18_Genetic/result_Gene2_Correlation/subtype1/subtype1_PLSscore.dscalar.nii"
]

# ============================================================
# 2. fsLR 32k Surface 文件
# ============================================================
template_dir = (
    "/Users/qingchen/Documents/Data/template/"
    "HCP_S1200_Atlas_Z4_pkXDZ"
)
left_surf = os.path.join(
    template_dir,
    "S1200.L.pial_MSMAll.32k_fs_LR.surf.gii"
)
right_surf = os.path.join(
    template_dir,
    "S1200.R.pial_MSMAll.32k_fs_LR.surf.gii"
)

# ============================================================
# 3. Brainnetome Atlas
# ============================================================
atlas_parcellation_file = (
    "/Users/qingchen/Documents/Data/template/"
    "BrainnetomeAtlas/BN_Atlas_freesurfer/fsaverage/"
    "fsaverage_LR32k/fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii"
)

# ============================================================
# 4. 背景阴影
# ============================================================
left_sulc = None
right_sulc = None

# ============================================================
# 5. 输出文件夹
# ============================================================
output_dir = ("/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_covariate/Step2_subtype/Step18_Genetic/result_Gene2_Correlation/subtype1/s1_PLSscoreMapBrain")
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 6. fsLR 32k 顶点数
# ============================================================
N_VERTICES_PER_HEMI = 32492

# ============================================================
# 7. 读取 Brainnetome Atlas
# ============================================================
print("\n============================================================")
print("Loading Brainnetome atlas...")
print("============================================================")
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
print("Brainnetome atlas loaded.")

# ============================================================
# 8. 全局统一原始数据颜色范围【两个subtype共用，不可每个图改】
# ============================================================
GLOBAL_VMIN = -1  # 你的最小值
GLOBAL_VMAX = 1  # 你的最大值

# ============================================================
# 9. 非线性颜色增强参数
# ============================================================
'''
gamma < 1：增强低值区域的颜色变化
gamma = 1：普通线性映射
gamma > 1：增强高值区域的颜色变化
推荐：gamma = 0.4~0.6
⚠️两个subtype必须用完全同一个gamma，保证颜色映射规则一致
'''
gamma = 0.4

# ============================================================
# 10. 自定义颜色
# ============================================================
my_colors = [
    "#00478D",
    "#5C86B3",
    "#9DB8D0",

    "#FAFDFF",  # 这是中心点 0.5 对应的白色

    "#E3A0AA",
    "#D15A6B",
    "#B50A27"
]
my_custom_cmap = LinearSegmentedColormap.from_list(
    "INT_Custom_Cmap",
    my_colors,
    N=256
)

# ============================================================
# 11. View
# ============================================================
views = {
    "Lateral": "lateral",
    "Medial": "medial"
}


# ============================================================
# 12. 定义【对称】非线性变换函数 (修复白色中心点偏移问题)
# ============================================================
def nonlinear_transform(data, vmin, vmax, gamma):
    """
    对称的非线性变换：
    1. 计算中心点，将数据从 [vmin, vmax] 映射到 [-1, 1] 范围
    2. 使用带有符号的 power 变换，以此保留0点（即原本的中心值，映射后代表白色）不变
    3. 重新将 [-1, 1] 映射回 [0, 1] 以匹配 matplotlib 的绘图区间
    NaN保持NaN，clip限制，消除警告
    """
    center = (vmin + vmax) / 2.0
    half_range = (vmax - vmin) / 2.0

    # 映射到 [-1, 1] 空间，中心点变为0
    data_norm = (data - center) / half_range
    data_norm = np.clip(data_norm, -1.0, 1.0)

    out = np.full_like(data_norm, np.nan)
    mask = np.isfinite(data_norm)

    # 对正负两端对称进行 gamma 幂运算，增强渐变同时保真符号
    out[mask] = np.sign(data_norm[mask]) * np.power(np.abs(data_norm[mask]), gamma)

    # 将 [-1, 1] 映射回 [0, 1]
    out = (out + 1.0) / 2.0
    return out


# ============================================================
# 13. 循环处理 dscalar
# ============================================================
all_file_stats = []
for file_idx, dscalar_file in enumerate(dscalar_files):
    print("\n")
    print("============================================================")
    print(f"Processing file {file_idx + 1}/{len(dscalar_files)}")
    print(os.path.basename(dscalar_file))
    print("============================================================")

    file_basename = os.path.basename(dscalar_file).split(".")[0]
    img = nib.load(dscalar_file)
    data = img.get_fdata()
    if data.shape[0] == 1:
        data = data[0]
    brain_model_axis = img.header.get_axis(1)

    left_full_data = np.full(N_VERTICES_PER_HEMI, np.nan)
    right_full_data = np.full(N_VERTICES_PER_HEMI, np.nan)

    for structure, slc, brain_model in brain_model_axis.iter_structures():
        structure_data = data[slc]
        if structure == "CIFTI_STRUCTURE_CORTEX_LEFT":
            left_full_data[brain_model.vertex] = structure_data
        elif structure == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            right_full_data[brain_model.vertex] = structure_data

    left_full_data[left_full_data == 0] = np.nan
    right_full_data[right_full_data == 0] = np.nan

    all_values = np.concatenate([
        left_full_data[np.isfinite(left_full_data)],
        right_full_data[np.isfinite(right_full_data)]
    ])
    v_data_min = np.min(all_values)
    v_data_max = np.max(all_values)
    print("\nActual data range:")
    print(f"    Minimum = {v_data_min:.4f}")
    print(f"    Maximum = {v_data_max:.4f}")
    print(f"    Mean    = {np.mean(all_values):.4f}")
    print(f"    Median  = {np.median(all_values):.4f}")
    all_file_stats.append((file_basename, v_data_min, v_data_max))

    # ===== 关键：两个subtype全部使用全局GLOBAL_VMIN/GLOBAL_VMAX做对称式非线性变换 =====
    left_plot_data = nonlinear_transform(left_full_data, GLOBAL_VMIN, GLOBAL_VMAX, gamma)
    right_plot_data = nonlinear_transform(right_full_data, GLOBAL_VMIN, GLOBAL_VMAX, gamma)

    plot_vmin = 0.0
    plot_vmax = 1.0

    hemispheres = {
        "Left": (left_surf, left_plot_data, left_sulc, "left", left_atlas_labels),
        "Right": (right_surf, right_plot_data, right_sulc, "right", right_atlas_labels)
    }

    for hemi_name, (surf_mesh, surf_map, bg_map, hemi_dir, hemi_atlas_labels) in hemispheres.items():
        if bg_map is None:
            bg_map = np.ones(surf_map.shape) * 0.1

        for view_name, view_ang in views.items():
            output_file = os.path.join(
                output_dir,
                f"{file_basename}_{hemi_name}_{view_name}_meanINT.png"
            )
            print(f"\nGenerating: {hemi_name} - {view_name}")

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
            plotting.plot_surf(
                surf_mesh=surf_mesh,
                surf_map=surf_map,
                hemi=hemi_dir,
                view=view_ang,
                bg_map=bg_map,
                cmap=my_custom_cmap,
                vmin=plot_vmin,
                vmax=plot_vmax,
                colorbar=False,  # 全部脑图关闭colorbar，统一使用外部一张共享colorbar
                title=f"{file_basename}\n{hemi_name} Hemisphere - {view_name}",
                axes=ax
            )
            plt.savefig(output_file, dpi=300, bbox_inches="tight", transparent=True)
            print("Saved:", output_file)
            plt.close(fig)

# ============================================================
# 18. 制作公共 Colorbar —— 所有图共用这同一个colorbar
# ============================================================
print("\n")
print("============================================================")
print("Generating shared global colorbar...")
print("============================================================")
for name, dmin, dmax in all_file_stats:
    print(f"{name}: data [{dmin:.4f}, {dmax:.4f}]")
print(f"Global mapping range: [{GLOBAL_VMIN:.2f}, {GLOBAL_VMAX:.2f}], gamma={gamma}")

fig = plt.figure(figsize=(8, 1.5))
ax = fig.add_axes([0.10, 0.40, 0.80, 0.25])

norm = Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap=my_custom_cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=ax, orientation="horizontal")

# 原始物理刻度（取中心点及左右对称分布的刻度）
original_ticks = np.linspace(GLOBAL_VMIN, GLOBAL_VMAX, 5)  # 5个刻度能更好地确保中间刻度正是(VMAX+VMIN)/2
# 使用与绘图数据相同的非线性变换来定位刻度！这样刻度位置自动贴合色彩变化
transformed_ticks = nonlinear_transform(original_ticks, GLOBAL_VMIN, GLOBAL_VMAX, gamma)

cbar.set_ticks(transformed_ticks)
cbar.set_ticklabels([f"{x:.2f}" for x in original_ticks])

cbar.ax.tick_params(labelsize=12, length=5, width=1)
cbar.outline.set_linewidth(1)

colorbar_file = os.path.join(output_dir, "Shared_Colorbar.png")
plt.savefig(colorbar_file, dpi=300, bbox_inches="tight", transparent=True)
print("Saved shared colorbar:", colorbar_file)
plt.close(fig)

print("\n")
print("============================================================")
print("All processing completed!")
print("============================================================")
print(f"\nOutput directory:\n{output_dir}")
print(f"\nGlobal value range: {GLOBAL_VMIN:.2f} → {GLOBAL_VMAX:.2f}")
print(f"\nGamma symmetric power transform: {gamma}")
print("\nColor mapping: Blue → Light Blue → White → Light Red → Red")