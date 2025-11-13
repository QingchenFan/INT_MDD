import os
import pandas as pd
import pcntoolkit as ptk
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

HC_train = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC_MDD/INT_NM/Feature/HC_Train_harmonized.csv')
                # 获取其他站点名称
HC_discover = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC_MDD/INT_NM/Feature/HC_DiscoverSet_harmonized.csv')


pro_dir = '/Volumes/QC/INT/INT_BN246_HC_MDD/INT_NM/Results/INT246_GPR_20251112/NMResults/'
if not os.path.isdir(pro_dir):
    os.mkdir(pro_dir)
os.chdir(pro_dir)
pro_dir = os.getcwd()

brainRegion = HC_train.columns.tolist()
idps = brainRegion[4:]

#  ---训练数据集---
X_train = (HC_train[['sex', 'age']]).to_numpy(dtype=float)
Y_train = HC_train[idps].to_numpy(dtype=float)

with open('X_train.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_train), file)
with open('Y_train.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_train), file)
X_Train_covfile = os.path.join(pro_dir, 'X_train.pkl')  # covariates (eg age) the training samples (columns: covariates, rows: observations or subjects)
Y_Train_respfile = os.path.join(pro_dir, 'Y_train.pkl')

#  ---发现数据集---
X_discover_covfile = (HC_train[['sex', 'age']]).to_numpy(dtype=float)
Y_discover_covfile = HC_train[idps].to_numpy(dtype=float)
with open('X_discover.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_discover_covfile), file)
with open('Y_discover.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_discover_covfile), file)
Discover_covfile = os.path.join(pro_dir, 'X_discover.pkl')  # covariates (eg age) the training samples (columns: covariates, rows: observations or subjects)
Discover_respfile = os.path.join(pro_dir, 'Y_discover.pkl')  # measurements  (eg cortical thickness) of the training samples (columns: the various features/ROIs, rows: observations or subjects)

#  ---常模---
output_path = os.path.join(pro_dir, 'Models/')  # output path, where the models will be written
log_dir = os.path.join(pro_dir, 'log/')
outputsuffix = 'Test'
if not os.path.isdir(output_path):
    os.mkdir(output_path)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
ptk.normative.estimate(covfile=X_Train_covfile,
                       respfile=Y_Train_respfile,
                       testcov=Discover_covfile,
                       testresp=Discover_respfile,
                       cvfolds=None,
                       alg='gpr',
                       log_path=log_dir,
                       output_path=output_path,
                       outputsuffix=outputsuffix,
                       savemodel=True)