import glob
import os
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiMasker
from scipy.io import savemat
from nilearn import plotting
def volume_from_cifti(data, axis):
    assert isinstance(axis, nib.cifti2.BrainModelAxis)
    data = data.T[axis.volume_mask]                          # Assume brainmodels axis is last, move it to front
    volmask = axis.volume_mask                               # Which indices on this axis are for voxels?
    vox_indices = tuple(axis.voxel[axis.volume_mask].T)      # ([x0, x1, ...], [y0, ...], [z0, ...])
    vol_data = np.zeros(axis.volume_shape + data.shape[1:],  # Volume + any extra dimensions
                        dtype=data.dtype)
    vol_data[vox_indices] = data                             # "Fancy indexing"
    return nib.Nifti1Image(vol_data, axis.affine)

def surf_data_from_cifti(data, axis, surf_name):
    assert isinstance(axis, nib.cifti2.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():  # Iterates over volumetric and surface structures
        if name == surf_name:                                 # Just looking for a surface
            data = data.T[data_indices]                       # Assume brainmodels axis is last, move it to front
            vtx_indices = model.vertex                        # Generally 1-N, except medial wall vertices
            surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
            surf_data[vtx_indices] = data
            return surf_data
    raise ValueError(f"No structure named {surf_name}")

def loadData(datapath):
    cifti = nib.load(datapath)
    cifti_data = cifti.get_fdata()
    cifti_hdr = cifti.header
    axes = [cifti_hdr.get_axis(i) for i in range(cifti.ndim)]
    return cifti,cifti_data, cifti_hdr, axes

def subc_timeseries(data, atlaspath):
    data = data.transpose(3, 0, 1, 2)
    atlasData = nib.load(atlaspath).get_fdata()

    atlasData = np.reshape(atlasData, (1, 91 * 109 * 91), order='F')

    data = np.reshape(data, (data.shape[0], 91 * 109 * 91), order='F')

    roilist = []
    for r in range(211, 247):

        index = np.where(atlasData == r)

        roi = data[:, index[1]]  # 将第r个脑区中的voxel 数据（时间序列）提取

        # 统计体素个数
        totalvoxel = roi.shape[1] if roi.shape[1] > 0 else 1

        sum = np.sum(roi, axis=1)

        roiBoldsum = sum / totalvoxel

        roilist.append(roiBoldsum)

    subctimeseries = np.array(roilist)
    print('subctimeseries.shape-', subctimeseries.shape)
    subcFC = np.corrcoef(subctimeseries)
    return subcFC, subctimeseries

def calculate_FC(datapath, tpath, regions, atlaspath):
    #datapath = '/Users/qingchen/Documents/code/Data/FC/sub-06202_task-rest_space-fsLR_den-91k_desc-denoisedSmoothed_bold.dtseries.nii'

    cifti, cifti_data, cifti_hdr, axes = loadData(datapath)
    axes = [cifti_hdr.get_axis(i) for i in range(cifti.ndim)]

    Subcortical_Data = volume_from_cifti(cifti_data, axes[1])
    Subcortical_Data = Subcortical_Data.get_fdata()
    _, subctimeseries = subc_timeseries(Subcortical_Data, atlaspath)

    a_left = surf_data_from_cifti(cifti_data, axes[1], 'CIFTI_STRUCTURE_CORTEX_LEFT')
    a_right = surf_data_from_cifti(cifti_data, axes[1], 'CIFTI_STRUCTURE_CORTEX_RIGHT')
    cortex_data = np.concatenate((a_left, a_right), axis=0)
    print('cortex_data', cortex_data.shape)
    template = tpath
    label = nib.load(template).get_fdata()
    print('Label:', label.shape)
    roilist = []
    for i in range(1, regions + 1):
        index = np.where(label == i)
        roi = cortex_data[index[1], :]
        roilist.append(np.mean(roi, axis=0))

    roiMatrix = np.array(roilist)
    timeseries = np.vstack((roiMatrix, subctimeseries))

    return timeseries


# --- Test---未完成的代码
datapath = '/Volumes/QCI/NormativeModel/Data135/MDD/dtseriesnii/*.dtseries.nii'
box = glob.glob(datapath)

template = '/Users/qingchen/Documents/Data/template/CBIG-master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/Schaefer2018_400Parcels_7Networks_order.dlabel.nii'
atlaspath = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_246_2mm.nii.gz'

for i in box:
    # subID = i.split('/')[-2]
    # print(subID)
    subID = i.split('/')[-1][0:10]
    print(subID)

    timeseries = calculate_FC(i, template, 400, atlaspath)
    timeseries = timeseries.T
    print(timeseries.shape)
    newpath = "/Volumes/QC/Data/Schaefer400_BN36/Data135_MDD/" + subID
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    newdatap = newpath + '/' + subID + '.txt'


    np.savetxt(newdatap, timeseries)




