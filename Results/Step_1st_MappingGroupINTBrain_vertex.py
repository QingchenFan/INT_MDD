import glob
import numpy as np
import nibabel as nib
import pandas as pd
from nibabel import cifti2
import scipy.io as sio
path = "/Volumes/QC/NormativeModel/Data135/HC/INT/Group/*"
databox = glob.glob(path)

box = np.empty((1, 91282))
for i,j in enumerate(databox):
    start_index = i * 1000
    end_index = (i + 1) * 1000
    #print(j, '|', 'start:', start_index, '--end:', end_index)
    data = sio.loadmat(j)['hwhm']

    #print('data shape-',data.shape)
    if data.shape[1] != 1282:

        box[:, start_index:end_index] = data
    else:

        box[:, 90000:] = data

new_data = box
sio.savemat("lGroup.mat",{'l':new_data})
l_box = np.zeros((1, 32492))

ind_l = np.loadtxt('/Users/qingchen/Documents/Data/template/metric_index_L.txt').reshape(1,-1)
L_brain = new_data[:, 0:29696]
l_box[:, ind_l.astype(int)] = L_brain

r_box = np.zeros((1, 32492))
ind_r = np.loadtxt('/Users/qingchen/Documents/Data/template/metric_index_R.txt').reshape(1,-1)
R_brain = new_data[:, 29696:59412]
r_box[:, ind_r.astype(int)] = R_brain

resdata = np.concatenate((l_box,r_box), axis=1)
#print(new_data)
print(resdata.shape)

# ----------------------------------------------------------------
atlas = nib.load("/Users/qingchen/Documents/code/Data/FC/Schaefer2018_400Parcels_17Networks_order.dscalar.nii")
adata = atlas.get_fdata()

sio.savemat("schaefer400.mat",{'s':adata})
scalar_axis=nib.cifti2.cifti2_axes.ScalarAxis(['val'])
brain_model_axis=atlas.header.get_axis(1)

val_head=nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_model_axis))
nib.Cifti2Image(resdata, val_head, atlas.nifti_header, atlas.extra, atlas.file_map).to_filename('./INT_Data135.dscalar.nii')


# ------------Tool  --------------------------------
# 加载已有的 .dtseries.nii 文件
# existing_file_path = '/Volumes/QCI/NormativeModel/Data135/HC/dtseriesnii/sub-HC001_task-rest_acq-ap_run-1_space-fsLR_den-91k_desc-denoisedSmoothed_bold.dtseries.nii'
# cifti_img = nib.load(existing_file_path)
#
# # 读取原数据并获取形状
# original_data = cifti_img.get_fdata()
# print("Original data shape:", original_data.shape)
#
# # # 检查新数据的形状是否与原始数据匹配
# # if new_data.shape != original_data.shape:
# #     raise ValueError("新数据矩阵的形状与原文件不匹配！")
#
# # 替换原文件中的数据
# cifti_img = nib.Cifti2Image(new_data, header=cifti_img.header)
#
# # 保存为新的 .dtseries.nii 文件
# output_file_path = './output_replaced.dtseries.nii'
# nib.save(cifti_img, output_file_path)
#
# print(f"已将新数据矩阵保存到 {output_file_path}")

# ------------Tool  --------------------------------