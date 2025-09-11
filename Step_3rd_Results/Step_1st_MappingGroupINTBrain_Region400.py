import glob
import os
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiMasker
from scipy.io import savemat
from nilearn import plotting
import scipy.io as sio

tpath = '/Users/qingchen/Documents/Data/template/CBIG-master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/Schaefer2018_400Parcels_7Networks_order.dlabel.nii'
template = tpath
template = nib.load(template)
label=template.get_fdata()
print(label.shape)

data = sio.loadmat("/Volumes/QC/INT/INT_BN246_HC135BP_BP135MDD/Schaefer436_surface/INT_HCGroup/Group.mat")['hwhm']
print(data.shape)

for i in range(1, 401):

    index = np.where(label == i)
    label[:, index] = data[:, i-1]


scalar_axis = nib.cifti2.cifti2_axes.ScalarAxis(['INTvalue'])
brain_model_axis = template.header.get_axis(1)
scalar_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_model_axis))
scalar_img = nib.Cifti2Image(label, header=scalar_header)
scalar_img.to_filename('/Volumes/QC/INT/INT_BN246_HC135BP_BP135MDD/Schaefer436_surface/INT_GroupHC.dscalar.nii')
