import nibabel as nib
import numpy as np
'''
    检查 xcpd 输出的surface 数据和 BNA surface label 是否空间位置一致
'''
bold_path = (
    '/Users/qingchen/Desktop/sub-01000008V01_task-rest_acq-ap_run-1_space-fsLR_den-91k_desc-denoisedSmoothed_bold.dtseries.nii'

)

atlas_path = (
    '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/'
    'fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii'
)

# =========================
# 1. 读取 BOLD CIFTI
# =========================
bold = nib.load(bold_path)

print('========== BOLD ==========')
print('shape:', bold.shape)

for i in range(bold.ndim):
    axis = bold.header.get_axis(i)
    print('\nAxis', i)
    print(type(axis))

    if isinstance(axis, nib.cifti2.BrainModelAxis):
        print('BrainModelAxis')
        print('size:', axis.size)
        print('volume shape:', axis.volume_shape)

        for name, slc, model in axis.iter_structures():
            print(
                name,
                'slice =', slc,
                'size =', model.size,
                'type =', type(model).__name__
            )

# =========================
# 2. 读取 BN246 dlabel
# =========================
atlas = nib.load(atlas_path)

print('\n========== ATLAS ==========')
print('shape:', atlas.shape)

for i in range(atlas.ndim):
    axis = atlas.header.get_axis(i)

    print('\nAxis', i)
    print(type(axis))

    if isinstance(axis, nib.cifti2.BrainModelAxis):
        print('BrainModelAxis')
        print('size:', axis.size)

        for name, slc, model in axis.iter_structures():
            print(
                name,
                'slice =', slc,
                'size =', model.size,
                'type =', type(model).__name__
            )

# =========================
# 3. 读取 atlas label
# =========================
label = atlas.get_fdata()

print('\n========== LABEL ==========')
print('label shape:', label.shape)

unique = np.unique(label)

print('number of unique labels:', len(unique))
print('minimum:', unique.min())
print('maximum:', unique.max())

print('first labels:', unique[:20])
print('last labels:', unique[-20:])

bold_axis = bold.header.get_axis(1)
atlas_axis = atlas.header.get_axis(1)

for structure in [
    'CIFTI_STRUCTURE_CORTEX_LEFT',
    'CIFTI_STRUCTURE_CORTEX_RIGHT'
]:
    b = bold_axis[bold_axis.name == structure]
    a = atlas_axis[atlas_axis.name == structure]

    print(structure)
    print("BOLD size :", b.size)
    print("Atlas size:", a.size)
    print("vertex identical:", np.array_equal(b.vertex, a.vertex))