import glob
import numpy as np
import scipy.io as scio
from brainspace.gradient import GradientMaps
from scipy.io import savemat


path = '/Volumes/QC/Data/BN246_FC/HC_BP135/sub-*.mat'
#path = '/Volumes/QC/Data/BN246_FC/MDD_BP35/sub-*.mat'
dataList = glob.glob(path)

databox = []
box = np.zeros([246, 246])

for i in dataList:
    data = scio.loadmat(i)['data']
    data = np.clip(data, -0.99999, 0.99999)  # 截断处理
    fizFC = np.arctanh(data)
    box = np.add(box, fizFC)

mfizFC = box / len(dataList)
mFC = np.tanh(mfizFC)

gp = GradientMaps(kernel='normalized_angle', approach='dm', alignment='procrustes', n_components=10,
                  random_state=0)
# TODO: 计算MDD组梯度时，可向HC组梯度对齐
# ref = scio.loadmat('/Volumes/QC/INT/INT_BN246_HC135BP_BP135MDD/Gradient_HCgroup/BP135_HC_GroupGradient.mat')
# gp.fit(mFC, reference=ref['data'])

gp.fit(mFC)

res = gp.gradients_

savemat('/Volumes/QC/INT/INT_BN246_HC135BP_BP135MDD/Gradient_MDDgroup/BP135_HC_GroupGradient2.mat', {'data': res})

# # Plot brain gradient
# labeling = load_parcellation('schaefer', scale=400, join=True)
# surf_lh, surf_rh = load_conte69()
# mask = labeling != 0
#
# grad = [None] * 2
#
# for i in range(2):
#     # map the gradient to the parcels
#     grad[i] = map_to_labels(res[:, i], labeling, mask=mask, fill=np.nan)
#
# plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(2000, 800), cmap='coolwarm',
#                  color_bar=True, label_text=['Grad1', 'Grad2'], zoom=1)
#
# fig, ax = plt.subplots(1, figsize=(5, 4))
# ax.scatter(range(gp.lambdas_.size), gp.lambdas_)
# ax.set_xlabel('Component Nb')
# ax.set_ylabel('Eigenvalue')
#
# plt.show()