import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct
from sklearn.gaussian_process.kernels import ConstantKernel as C
Data = pd.read_csv("/Volumes/QC/INT/INT_BN246_HC135BP_allMDD/Results/INTvalue_HCMDD.csv")

region = pd.read_csv("/Volumes/QC/INT/INT_BN246_HC135BP_allMDD/Results/signifRegion.csv")
regions = region.columns.tolist()
MDDData = Data[Data['disorder'] == 1].sample(n=222)
HCData = Data[Data['disorder'] == 0]

Data = pd.concat([HCData, MDDData])

x_data = np.array(Data[regions])

y_label = np.array(Data['disorder'])

kf = KFold(n_splits=5,shuffle=True,random_state=6)
acc_res = []
kappa_res = []
for train_index, test_index in kf.split(x_data):
    # split data
    X_train, X_test = x_data[train_index, :], x_data[test_index, :]
    y_train, y_test = y_label[train_index], y_label[test_index]

    kernel = C(0.1, (1e-5, np.inf)) * DotProduct(sigma_0=0.1) ** 2
    gp = GaussianProcessClassifier(kernel=kernel)
    gp.fit(X_train, y_train)
    print("Learned kernel: %s " % gp.kernel_)
    Predict_Score = gp.predict(X_test)
    y_predict_proba = gp.predict_proba(X_test)

    acc = accuracy_score(y_test, Predict_Score)
    print('-acc = %.2f:' %(acc))
    acc_res.append(float("%.2f"%(acc)))

    kappa = cohen_kappa_score(np.array(y_test).reshape(-1, 1), np.array(Predict_Score).reshape(-1, 1))
    print('-kappa = %.2f:' % (kappa))
    kappa_res.append(kappa)
print('Result: acc=%.3f, kappa=%.3f ' % (np.mean(acc_res), np.mean(kappa_res)))
