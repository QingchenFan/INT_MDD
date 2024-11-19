import nibabel as nib

# import nibabel as nib
# import numpy as np
#
# # 读取.nii.gz文件
# nii_file = '/Users/qingchen/Desktop/testdata/sub-HC001/sub-HC001_task-rest_acq-ap_run-1_space-MNI152NLin6Asym_res-2_desc-denoisedSmoothed_bold.nii.gz'  # 替换为你的文件路径
# img = nib.load(nii_file)
# data = img.get_fdata()
#
# # 确保数据维度是91x109x91x200
# print("Original data shape:", data.shape)
#
# # 将维度变换成902629x200
# # 首先计算体素总数
# num_voxels = data.shape[0] * data.shape[1] * data.shape[2]
# reshaped_data = data.reshape(data.shape[3],num_voxels)
# print(reshaped_data.shape)
# # 保存到txt文件
# np.savetxt('./sub-HC001.txt', reshaped_data)

import nibabel as nib
import numpy as np

# 读取.nii.gz文件
nii_file = '/Users/qingchen/Documents/Notes/datalearn/lianxi/bids/derivatives/xcp_abcd/sub-001/func/sub-001_task-rest_space-fsLR_den-91k_desc-residual_smooth_den-91k_bold.dtseries.nii'  # 替换为你的文件路径
img = nib.load(nii_file)
data = img.get_fdata()
data = data[:,0:1000]

print("Original data shape:", data.shape)

# 保存到txt文件
np.savetxt('./sub-HC005.txt', data)


# import nibabel as nib
# import numpy as np
#
# # 读取.nii.gz文件
# nii_file = '/Users/qingchen/Desktop/testdata/MNI152_T1_2mm_Brain_Mask.nii.gz'  # 替换为你的文件路径
# img = nib.load(nii_file)
# data = img.get_fdata()
#
# # 确保数据维度是91x109x91
# print("Original data shape:", data.shape)
#
# # 将维度变换成902629x1
# num_voxels = data.shape[0] * data.shape[1] * data.shape[2]
# reshaped_data = data.reshape(num_voxels, 1)
#
# # 保存到txt文件
# np.savetxt('./mask_data9.txt', reshaped_data, fmt='%.0f')

import numpy as np

# 创建一个长度为N的全1一维矩阵
N = 1000  # 替换为所需的长度
ones_array = np.ones(N)

# 保存到txt文件中
np.savetxt('mask1000.txt', ones_array, fmt='%d')

print("已保存到 ones_array.txt 文件中")