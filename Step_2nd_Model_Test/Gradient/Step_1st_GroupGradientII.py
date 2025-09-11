import glob
import numpy as np
import scipy.io as scio
import nibabel as nib
from brainspace.gradient import GradientMaps
from brainspace.utils.parcellation import map_to_labels
from scipy.io import savemat
from brainspace.datasets import load_parcellation, load_conte69
from brainspace.gradient import GradientMaps
from brainspace.plotting import plot_hemispheres
#----Step 1 -----
'''
计算数据格式为pconn.nii的梯度
'''
#path = '/Volumes/QC/Data/Schaefer456_FC/HC_BP135/*.pconn.nii'
path = '/Volumes/QC/Data/Schaefer456_FC/MDD_BP135/*.pconn.nii'
dataList = glob.glob(path)

databox = []
box = np.zeros([456, 456])

for i in dataList:
    data = nib.load(i).get_fdata()
    fizFC = np.arctanh(data)
    box = np.add(box, fizFC)

mfizFC = box / len(dataList)
mFC = np.tanh(mfizFC)
mFC = mFC[0:400, 0:400]

gp = GradientMaps(kernel='pearson', approach='dm' )  # {'pearson', 'spearman', 'cosine', 'normalized_angle', 'gaussian'}

# TODO: 计算MDD组梯度时，可向HC组梯度对齐
# ref = scio.loadmat('/Volumes/QC/INT/INT_BN246_HC135BP_BP135MDD/Gradient_HCgroup/Sch456/BP135_HC_GroupGradient.mat')
# gp.fit(mFC, reference=ref['data'])
# TODO: 计算HC梯度
gp.fit(mFC)

res = gp.gradients_

savemat('/Volumes/QC/INT/INT_BN246_HC135BP_BP135MDD/Gradient_MDDgroup/Sch456/BP135_MDD_GroupGradient.mat', {'data': res})
#----Step 2 -----
import matplotlib.pyplot as plt

expl_var = gp.lambdas_ / sum(gp.lambdas_)

plt.figure(figsize=(5, 4))
plt.scatter(range(expl_var.size), expl_var * 100, alpha=0.7, color='#00063F')
plt.xlabel('Gradient', fontsize=14, fontname='Avenir')
plt.ylabel('Explained variance (%)', fontsize=14, fontname='Avenir')
plt.xticks(np.arange(len(expl_var)), np.arange(1, len(expl_var) + 1))  # axis ticks start at 1 not 0
plt.savefig('./MDD.png', dpi=300)
#----Step 3 -----
# Plot brain gradient
labeling = load_parcellation('schaefer', scale=400, join=True)
surf_lh, surf_rh = load_conte69()
mask = labeling != 0

grad = [None] * 2

for i in range(2):
    # map the gradient to the parcels
    grad[i] = map_to_labels(res[:, i], labeling, mask=mask, fill=np.nan)

plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(2000, 800), cmap='coolwarm',
                 color_bar=True, label_text=['Grad1', 'Grad2'], zoom=1)

fig, ax = plt.subplots(1, figsize=(5, 4))
ax.scatter(range(gp.lambdas_.size), gp.lambdas_)
ax.set_xlabel('Component Nb')
ax.set_ylabel('Eigenvalue')

plt.show()