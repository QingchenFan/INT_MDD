
import glob
import os
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiMasker
from scipy.io import savemat
from nilearn import plotting
import scipy.io as sio

tpath = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii'
template = tpath
template = nib.load(template)
label=template.get_fdata()
label[label > 210] -= 210

data = sio.loadmat("/Volumes/QC/INT/INT_BN246_HC135BP_allMDD/INT_MDDgroup/Group.mat")['hwhm']


for i in range(1, data.shape[1]+1):
    index = np.where(label == i)
    label[:,index] = data[:,i-1]



scalar_axis = nib.cifti2.cifti2_axes.ScalarAxis(['meanZvalue'])
brain_model_axis = template.header.get_axis(1)
scalar_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_model_axis))
scalar_img = nib.Cifti2Image(label, header=scalar_header)
scalar_img.to_filename('./INT_allDataMDD.dscalar.nii')
