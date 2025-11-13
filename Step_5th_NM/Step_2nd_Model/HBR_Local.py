import os
import pandas as pd
import pcntoolkit as ptk
import numpy as np
import pickle
#
# # a simple function to quickly load pickle files
def ldpkl(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)
HC_train = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC_MDD/INT_NM/Feature/HC_Train_harmonized.csv')
                # 获取其他站点名称
HC_discover = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC_MDD/INT_NM/Feature/HC_DiscoverSet_harmonized.csv')
#  输出路径
pro_dir = '/Volumes/QC/INT/INT_BN246_HC_MDD/INT_NM/Results/INT246_GPR_20251112/NMResults/'
if not os.path.isdir(pro_dir):
    os.mkdir(pro_dir)
os.chdir(pro_dir)
pro_dir = os.getcwd()

brainRegion = HC_train.columns.tolist()
idps = brainRegion[4:]
print(idps)

os.chdir(pro_dir)
pro_dir = os.getcwd()
# #  ---构建训练集---
X_train = (HC_train['age']/100).to_numpy(dtype=float)
Y_train = HC_train[idps].to_numpy(dtype=float)
batch_effects_train = HC_train[['sex']].to_numpy(dtype=int)

with open('X_train.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_train), file)
with open('Y_train.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_train), file)
with open('trbefile.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(batch_effects_train), file)

respfile = os.path.join(pro_dir, 'Y_train.pkl')       # measurements  (eg cortical thickness) of the training samples (columns: the various features/ROIs, rows: observations or subjects)
covfile = os.path.join(pro_dir, 'X_train.pkl')
trbefile = os.path.join(pro_dir, 'trbefile.pkl')      # training batch effects file (eg scanner_id, gender)  (columns: the various batch effects, rows: observations or subjects)


#  ---构建测试集---
X_test = (HC_discover['age']/100).to_numpy(dtype=float)
Y_test = HC_discover[idps].to_numpy(dtype=float)

batch_effects_test = HC_discover[['sex']].to_numpy(dtype=int)
with open('X_test.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_test), file)
with open('Y_test.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_test), file)
with open('tsbefile.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(batch_effects_test), file)


# covariates (eg age) the training samples (columns: covariates, rows: observations or subjects)

testrespfile_path = os.path.join(pro_dir, 'Y_test.pkl')       # measurements  for the testing samples
testcovfile_path = os.path.join(pro_dir, 'X_test.pkl')        # covariate file for the testing samples

tsbefile = os.path.join(pro_dir, 'tsbefile.pkl')      # testing batch effects file

output_path = os.path.join(pro_dir, 'Models/')    #  output path, where the models will be written
#
log_dir = os.path.join(pro_dir, 'log/')
if not os.path.isdir(output_path):
    os.mkdir(output_path)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
outputsuffix = '_estimate'
ptk.normative.estimate(covfile=covfile,
                       respfile=respfile,
                       trbefile=trbefile,
                       testcov=testcovfile_path,
                       testresp=testrespfile_path,
                       tsbefile=tsbefile,
                       alg='hbr',
                       log_path=log_dir,
                       binary=True,
                       output_path=output_path,
                       outputsuffix=outputsuffix,
                       savemodel=True)





#  ---构建MDD测试集---
'''
allMDD = pd.read_csv('/Volumes/QCI/NormativeModel/Results/Result_GrayVol246_HBR_HCMDD_1227/Feature/'
                     'AllMDD_GrayVol_all246_III.csv')

X_mdd_test = (allMDD['age']/100).to_numpy(dtype=float)
Y_mdd_test = allMDD[idps].to_numpy(dtype=float)
print(X_mdd_test.shape)
print(Y_mdd_test.shape)

batch_effects_mdd_test = allMDD[['sitenum', 'sex']].to_numpy(dtype=int)
print('be---', batch_effects_mdd_test.shape)
with open('X_mdd_test.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_mdd_test), file)
with open('Y_mdd_test.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_mdd_test), file)
with open('tsbefile_mdd.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(batch_effects_mdd_test), file)

mddcovfile = os.path.join(pro_dir, 'X_mdd_test.pkl')
mddrespfile = os.path.join(pro_dir, 'Y_mdd_test.pkl')       # measurements  (eg cortical thickness) of the training samples (columns: the various features/ROIs, rows: observations or subjects)
mddtsbefile = os.path.join(pro_dir, 'tsbefile_mdd.pkl')      # testing batch effects file

#output_path = os.path.join(pro_dir, 'Models/')    #  output path, where the models will be written

log_dir = os.path.join(pro_dir, 'log/')
if not os.path.isdir(output_path):
    os.mkdir(output_path)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
mddoutputsuffix = '_mdd'
yhat_te, s2_te, Z = ptk.normative.predict(
                            covfile=mddcovfile,
                            respfile=mddrespfile,
                            tsbefile=mddtsbefile,
                            alg='hbr',
                            log_path=log_dir,
                            binary=True,
                            model_path=output_path,
                            outputsuffix=mddoutputsuffix,
                            savemodel=True
)
'''
