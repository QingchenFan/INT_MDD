
import numpy as np
import nibabel as nib

import scipy.io as sio
from scipy.stats import ttest_ind
HCData = sio.loadmat('/Volumes/QC/INT/INT_BN246_HC135_allMDD/INT_HCind/sub-HC114_Ind.mat')['hwhms']
MDDData = sio.loadmat('/Volumes/QC/INT/INT_BN246_HC135BP_allMDD/INT_MDDind/sub-MDD143V01_Ind.mat')['hwhms']
box = []
roi = {}
for i in range(0,246):
    print('ROI:', i+1)
    hcdata = HCData[i,:]
    mdddata = MDDData[i,:]

    t, p = ttest_ind(hcdata, mdddata)

    box.append(p)
    if p < 0.05:
        print('ROI:', i+1,' ','P-value:', p, ' ','T-value:', t)

pvalue = np.array(box)
np.savetxt('pvalue.txt',pvalue)

tpath = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii'
template = tpath
template = nib.load(template)
label=template.get_fdata()
label[label > 210] -= 210

data = pvalue

data = np.where(data > 0.05,np.nan,data)

for i in range(1, data.shape[0]+1):
    index = np.where(label == i)
    label[:,index] = data[i-1]



scalar_axis = nib.cifti2.cifti2_axes.ScalarAxis(['meanZvalue'])
brain_model_axis = template.header.get_axis(1)
scalar_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_model_axis))
scalar_img = nib.Cifti2Image(label, header=scalar_header)
scalar_img.to_filename('./Ttestp_DataBP135_allMDD.dscalar.nii')
