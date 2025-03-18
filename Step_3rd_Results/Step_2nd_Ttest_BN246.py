import numpy as np
import nibabel as nib
import pandas as pd
import scipy.io as sio
from scipy.stats import ttest_ind
import statsmodels.stats.multitest as smm

HCData = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_BP135MDD/Results/INTvalue_HC.csv')
MDDData = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_BP135MDD/Results/INTvalue_MDD.csv')

brainRegion = HCData.columns.tolist()
del brainRegion[:1]

HCdata = np.array(HCData[brainRegion])
MDDdata = np.array(MDDData[brainRegion])

box = []
roi = []
tvalue = []
for i in range(0, 246):
    print('ROI:', i+1)
    hcdata = HCdata[:, i]
    mdddata = MDDdata[:, i]

    t, p = ttest_ind(hcdata, mdddata)
    roi.append(brainRegion[i])
    tvalue.append(t)
    box.append(p)
    if p < 0.05:
        print('ROI:', i+1, ' ', 'P-value:', p, ' ', 'T-value:', t)
pvalue = np.array(box)
rejected, fdr_pvalue, _, _ = smm.multipletests(pvalue, alpha=0.05, method='fdr_bh')


# 创建DataFrame，将rvalue、pvalue、fdr_pvalue对应保存
result_df = pd.DataFrame({
    'ROI': roi,
    'tvalue': tvalue,
    'pvalue': pvalue,
    'fdr-pvalue': fdr_pvalue,

})

# 将结果保存到CSV文件中，可根据实际需求修改文件路径及文件名
result_df.to_csv('./Ttest_Region.csv', index=False)




tpath = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii'
template = tpath
template = nib.load(template)
label=template.get_fdata()
label[label > 210] -= 210

data = pvalue
data = fdr_pvalue

data = np.where(data > 0.05, np.nan, data)
print(data)
for i in range(1, data.shape[0]+1):
    index = np.where(label == i)
    label[:, index] = data[i-1]



scalar_axis = nib.cifti2.cifti2_axes.ScalarAxis(['IntValue'])
brain_model_axis = template.header.get_axis(1)
scalar_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_model_axis))
scalar_img = nib.Cifti2Image(label, header=scalar_header)
scalar_img.to_filename('./Ttestfdrp_HCBP135_BP135MDD.dscalar.nii')
