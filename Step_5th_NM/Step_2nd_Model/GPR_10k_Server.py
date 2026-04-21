import os
import pandas as pd
import pcntoolkit as ptk
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

HC_train = pd.read_csv('/n04dat/kkwang/HCP/xicang/INT_NM/Feature/HC_Train_harmonized.csv')
                # 获取其他站点名称
HC_discover = pd.read_csv('/n04dat/kkwang/HCP/xicang/INT_NM/Feature/HC_DiscoverSet_harmonized.csv')


pro_dir = '/n04dat/kkwang/HCP/xicang/INT_NM/Results/INT246_GPR_20251113/Train/'
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


#  ---常模---
output_path = os.path.join(pro_dir, 'Models/')  # output path, where the models will be written
log_dir = os.path.join(pro_dir, 'log/')
outputsuffix = 'Train'
if not os.path.isdir(output_path):
    os.mkdir(output_path)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
ptk.normative.estimate(covfile=X_Train_covfile,
                       respfile=Y_Train_respfile,
                       cvfolds=10,
                       alg='gpr',
                       log_path=log_dir,
                       output_path=output_path,
                       outputsuffix=outputsuffix,
                       savemodel=True)