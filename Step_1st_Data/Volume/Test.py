import glob
import os
from pathlib import Path
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

# 1. 加载 BNA 模板图谱
template_path = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_246_2mm.nii.gz'
bn_img = nib.load(template_path)

bold_files = glob.glob('/Volumes/T7 Shield/*.nii.gz')

for bold_path in bold_files:
    bold_p = Path(bold_path)
    # 提取被试名（根据你的实际命名规则截取）
    subID = bold_p.stem.split('.')[0]
    print(f"Processing: {subID}")

    # 2. 加载当前被试的 4D BOLD 数据
    bold_img = nib.load(bold_path)
    fdata = bold_img.get_fdata()  # 形状为 (X, Y, Z, T)
    n_timepoints = fdata.shape[3]

    # 3. 空间对齐与重采样（关键修复）：

    resampled_atlas = resample_from_to(
        bn_img, (bold_img.shape[:3], bold_img.affine), order=0
    )
    atlas_data = np.rint(resampled_atlas.get_fdata()).astype(int)

    # 4. 提取各脑区时间序列
    roi_timeseries = np.zeros((n_timepoints, 246))

    for r in range(1, 247):
        mask = atlas_data == r
        voxel_count = np.sum(mask)

        if voxel_count == 0:
            print(f"Warning: ROI {r} has 0 voxels in subject {subID}")
            roi_timeseries[:, r - 1] = np.nan
        else:

            roi_timeseries[:, r - 1] = np.mean(fdata[mask], axis=0)

    # 5. 保存结果
    save_dir = Path(f"/Volumes/T7 Shield/{subID}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file = save_dir / f"{subID}.txt"

    np.savetxt(save_file, roi_timeseries)
    print(f"Saved: {save_file}, shape: {roi_timeseries.shape}")