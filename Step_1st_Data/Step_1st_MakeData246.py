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
    return cifti, cifti_data, cifti_hdr, axes

def plot_correlation_matrix(correlation_matrix,labels):
    plotting.plot_matrix(
        correlation_matrix,
        figure=(10, 8),
        labels=labels,
        vmax=0.8,
        vmin=-0.8,
        #title="Confounds",
        reorder=True,
    )
    plotting.show()
def subc_timeseries(data, atlaspath, brain_axis):
    data = data.transpose(3, 0, 1, 2)
    atlasData = nib.load(atlaspath).get_fdata()

    # 保留固定的标准空间尺寸
    if data.shape[1:] != (91, 109, 91):
        raise ValueError(f"数据空间尺寸错误：{data.shape[1:]}")
    if atlasData.shape != (91, 109, 91):
        raise ValueError(f"图谱空间尺寸错误：{atlasData.shape}")

    # CIFTI 实际收录的体素掩膜
    valid_mask = np.zeros((91, 109, 91), dtype=bool)
    vox_indices = tuple(
        brain_axis.voxel[brain_axis.volume_mask].T
    )
    valid_mask[vox_indices] = True

    # 三者使用相同的展平顺序，保证位置对应
    atlasData = np.reshape(
        atlasData, (1, 91 * 109 * 91), order='F'
    )
    data = np.reshape(
        data, (data.shape[0], 91 * 109 * 91), order='F'
    )
    valid_mask = np.reshape(
        valid_mask, (1, 91 * 109 * 91), order='F'
    )
    roilist = []
    for r in range(211, 247):
        # 同时属于该脑区，并且被 CIFTI 收录
        index = np.where((atlasData == r) & valid_mask)
        roi = data[:, index[1]]

        total = np.count_nonzero(atlasData == r)
        covered = roi.shape[1]
        coverage = covered / total if total else 0.0
        print(
            f"ROI {r}: 有效体素 {covered}/{total}, "
            f"覆盖率 {coverage:.1%}"
        )
        if covered == 0:
            raise ValueError(f"ROI {r} 没有有效体素")
        roiBoldmean = np.mean(roi, axis=1)
        roilist.append(roiBoldmean)
    subctimeseries = np.array(roilist)
    print('subctimeseries.shape-', subctimeseries.shape)

    subcFC = np.corrcoef(subctimeseries)
    return subcFC, subctimeseries


def calculate_FC(dpath, tpath, atlaspath, regions):
    datapath = dpath
    cifti, cifti_data, cifti_hdr, axes = loadData(datapath)
    axes = [cifti_hdr.get_axis(i) for i in range(cifti.ndim)]
    print("source data shape :", cifti_data.shape)

    Subcortical_Data = volume_from_cifti(cifti_data, axes[1])
    Subcortical_Data = Subcortical_Data.get_fdata()
    _, subctimeseries = subc_timeseries(Subcortical_Data, atlaspath,axes[1])
    # savemat('./subcFC.mat', {'data': subcFC})
    # savemat('./subctimeseries.mat', {'data': subctimeseries})

    a_left = surf_data_from_cifti(cifti_data, axes[1], 'CIFTI_STRUCTURE_CORTEX_LEFT')
    a_right = surf_data_from_cifti(cifti_data, axes[1], 'CIFTI_STRUCTURE_CORTEX_RIGHT')
    cortex_data = np.concatenate((a_left, a_right), axis=0)
    #print('cortex_data',cortex_data.shape)
    template = tpath
    label = nib.load(template).get_fdata()
    label[label > 210] -= 210

    if label.shape[1] == 59412:
        cortex_data = nib.load(datapath).get_fdata()[:, 0:59412].T
    roilist = []
    for i in range(1, regions + 1):
        index = np.where(label == i)
        roi = cortex_data[index[1], :]
        roilist.append(np.mean(roi, axis=0))

    roiMatrix = np.array(roilist)
    vertexsubc = np.vstack((roiMatrix, subctimeseries))
    resFC = np.corrcoef(vertexsubc)
    print('FC shape : ', resFC.shape)
    return resFC, vertexsubc
if __name__ == '__main__':
    datapath = '/Volumes/QC/Data/INT/brianproject/HC/sub-*_task-rest_acq-ap_run-1_space-fsLR_den-91k_bold_smooth.dtseries.nii'
    tpath = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii'
    atlaspath = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_246_2mm.nii.gz'
    data = glob.glob(datapath)

    for i in data:
        subID = i.split('/')[-1].split('_')[0]
        print(subID)

        newpath = "/Volumes/QC/Data/INT/BN246timeseries_surface/HC/" + subID
        if os.path.exists(newpath):
            continue
        resFC, vertexsubc = calculate_FC(i, tpath, atlaspath, 210)
        vertexsubc = vertexsubc.T
        print('vertexsubc', vertexsubc.shape)


        if not os.path.exists(newpath):
            os.makedirs(newpath)
        newdatap = newpath + '/' + subID + '.txt'


        np.savetxt(newdatap, vertexsubc)




