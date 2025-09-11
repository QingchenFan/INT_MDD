import nibabel as nib
import os
import numpy as np
import glob
from scipy.io import savemat

template = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_246_2mm.nii.gz'
BNData = nib.load(template)
templateData = BNData.get_fdata()
templateData = np.reshape(templateData, (1, 91 * 109 * 91), order='F')
#savemat('./templateData.mat', {'data': templateData})

boldpath = '/Volumes/QC/Data/VolumeData/Data135/MDD/*/*.nii.gz'
bolddata = glob.glob(boldpath)
for i in bolddata:
    subID = i.split('/')[-2]
    print(subID)
    fdata = nib.load(i).get_fdata()
    fdata =fdata.transpose(3, 0, 1, 2)
    fdata = np.reshape(fdata, (fdata.shape[0], 91 * 109 * 91), order='F')

    roilist = []
    for r in range(1, 247):
        index = np.where(templateData == r)
        roi = fdata[:, index[1]]  # 将第r个脑区中的voxel 数据（时间序列）提取

        totalvoxel = roi.shape[1]    # 统计体素个数

        sum = np.sum(roi, axis=1)
        roiBoldsum = sum / totalvoxel

        roilist.append(roiBoldsum)
    newpath = "/Volumes/QC/Data/VolumeData/Data135/MDD_BN246timeseries/" + subID
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    newdatap = newpath + '/' + subID + '.txt'
    roiMatrix = np.array(roilist)
    roiMatrix = roiMatrix.T
    print('roiMatrix-', roiMatrix.shape)
    np.savetxt(newdatap, roiMatrix)



