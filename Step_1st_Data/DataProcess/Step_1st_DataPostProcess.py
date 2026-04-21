import os
import nibabel as nib
import numpy as np
import subprocess
from nilearn.signal import clean
from nilearn.interfaces.fmriprep import load_confounds
from scipy.interpolate import interp1d

# ====================== 0. 路径配置 ======================
bold_cifti = "/Volumes/ZLabData/BrainProject/brainproject_I/fmriprep_out_HC/sub-01000008V01/func/sub-01000008V01_task-rest_acq-ap_run-1_space-fsLR_den-91k_bold.dtseries.nii"
# 使用绝对路径以确保 wb_command 运行安全
out_file = os.path.abspath("./sub-01000008V01_task-rest_acq-ap_run-1_space-fsLR_den-91k_bold_cleaned2.dtseries.nii")
smoothed_file = os.path.abspath("./sub-01000008V01_task-rest_acq-ap_run-1_space-fsLR_den-91k_bold_smooth.dtseries.nii")

# 表面文件路径 (确保与 91k 数据匹配的 32k 表面)
left_surf = "/Users/qingchen/Documents/Data/template/HCP_S1200_Atlas_Z4_pkXDZ/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
right_surf = "/Users/qingchen/Documents/Data/template/HCP_S1200_Atlas_Z4_pkXDZ/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii"

# ====================== 1. 加载 Confounds & 获取 Mask ======================
print("Loading confounds and calculating sample mask...")
confounds, sample_mask = load_confounds(
    img_files=bold_cifti,
    strategy=('motion', 'wm_csf', 'global_signal', 'high_pass'),
    motion='full',
    wm_csf='full',
    global_signal='full',
    scrub=True,
    fd_threshold=0.5,
    demean=True,
)

# ====================== 2. 加载 CIFTI 数据 ======================
img = nib.load(bold_cifti)
data = img.get_fdata()  # 形状通常为 (Time, Nodes)
n_timepoints, n_features = data.shape

# ====================== 3. 坏帧线性插值 (Censored Frames) ======================
if sample_mask is not None and len(sample_mask) < n_timepoints:
    print(f"Detected {n_timepoints - len(sample_mask)} frames with FD > 0.5. Interpolating...")

    all_frames = np.arange(n_timepoints)
    bad_frames = np.setdiff1d(all_frames, sample_mask)

    # 对每一列（顶点/体素）进行线性插值
    f_interp = interp1d(sample_mask, data[sample_mask, :], axis=0,
                        kind='linear', fill_value="extrapolate")

    data[bad_frames, :] = f_interp(bad_frames)
else:
    print("No frames exceeded FD threshold or all frames are valid.")

# ====================== 4. 信号清洗 (Nilearn Clean) ======================
print("Cleaning signal (Detrending, Standardizing, Filtering)...")
cleaned = clean(
    data,
    confounds=confounds,
    t_r=2.0,
    detrend=True,
    standardize=True,
    low_pass=0.1,
    #high_pass=0.01,
)

# 保存清洗后的中间文件
new_img = nib.Cifti2Image(
    dataobj=cleaned.astype(np.float32),
    header=img.header,
    nifti_header=img.nifti_header
)
nib.save(new_img, out_file)
print(f"Saved cleaned file to: {out_file}")

# ====================== 5. 运行 Workbench 平滑 ======================
print("Starting wb_command cifti-smoothing...")

# 这里的 4.25 是 Sigma，对应约 10mm FWHM
# 如果你想要 6mm FWHM，请将 4.25 改为 2.55

surf_sigma = "4.25"
vol_sigma = "2.55"
cmd = [
    "wb_command",
    "-cifti-smoothing",
    out_file,
    surf_sigma,  # Surface Sigma
    vol_sigma,  # Volume Sigma
    "COLUMN",
    smoothed_file,
    "-left-surface", left_surf,
    "-right-surface", right_surf
]


result = subprocess.run(cmd, check=True, capture_output=True, text=True)
