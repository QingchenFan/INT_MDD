import os
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
from pcntoolkit.normative import estimate, predict, evaluate
from pcntoolkit.util.utils import compute_MSLL, create_design_matrix
# from nm_utils import calibration_descriptives, remove_bad_subjects, load_2d


model_name = 'volume_246'

allHC = pd.read_csv('INT_HC.csv')
                # 获取其他站点名称
tr = np.random.uniform(size=allHC.shape[0]) > 0.2  # 形成一个随机抽样
te = ~tr

df_tr = allHC.loc[tr]
df_te = allHC.loc[te]                            # 将fcon中数据一分为2 ture false

#df_tr = pd.read_csv("/n04dat/kkwang/HCP/xicang/blr/Result_GrayVol246_BLR_HCMDD_250512/NMResults/allHC_tr.csv")
#df_te = pd.read_csv("/n04dat/kkwang/HCP/xicang/blr/Result_GrayVol246_BLR_HCMDD_250512/NMResults/allHC_te.csv")
print(" -Data Done ! -")
#TODO:
processing_dir = "/n04dat/kkwang/HCP/xicang/blr/Result_GrayVol246_BLR_HCMDD_250512/NMResults/"


if not os.path.isdir(processing_dir):
    os.mkdir(processing_dir)
df_tr.to_csv(processing_dir + '/allHC_tr.csv')
df_te.to_csv(processing_dir + '/allHC_te.csv')



# extract a list of unique site ids from the training set
site_ids = sorted(set(df_tr['sitenum'].to_list()))
print(site_ids)
site_test = sorted(set(df_te['sitenum'].to_list()))
print(site_test)

brainRegion = allHC.columns.tolist()
idp_ids = brainRegion[6:]
print(idp_ids)


# which data columns do we wish to use as covariates?
cols_cov = ['age', 'sex', 'TIV']#,'ICV_total_volume_cm3'

# which warping function to use? We can set this to None in order to fit a vanilla Gaussian noise model
warp = 'WarpSinArcsinh'

# limits for cubic B-spline basis
xmin = 18
xmax = 40

# Do we want to force the model to be refit every time?
force_refit = True

# Absolute Z treshold above which a sample is considered to be an outlier (without fitting any model)
outlier_thresh = 7

for idp_num, idp in enumerate(idp_ids):
    print('Running IDP', idp_num, idp, ':')

    # set output dir
    idp_dir = os.path.join(processing_dir, model_name, idp)
    os.makedirs(os.path.join(idp_dir), exist_ok=True)
    os.chdir(idp_dir)
    print(idp_dir)

    # extract the response variables for training and test set
    y_tr = df_tr[idp].to_numpy()

    y_te = df_te[idp].to_numpy()


    # write out the response variables for training and test
    resp_file_tr = os.path.join(idp_dir, 'resp_tr.txt')
    resp_file_te = os.path.join(idp_dir, 'resp_te.txt')
    np.savetxt(resp_file_tr, y_tr)
    np.savetxt(resp_file_te, y_te)
    print(df_te[cols_cov])

    # configure the design matrix
    X_tr = create_design_matrix(df_tr[cols_cov],
                                site_ids=df_tr['sitenum'],
                                basis='bspline',
                                xmin=xmin,
                                xmax=xmax)
    X_te = create_design_matrix(df_te[cols_cov],
                                site_ids=df_te['sitenum'],
                                all_sites=site_ids,
                                basis='bspline',
                                xmin=xmin,
                                xmax=xmax)

    # configure and save the covariates
    cov_file_tr = os.path.join(idp_dir, 'cov_bspline_tr.txt')
    cov_file_te = os.path.join(idp_dir, 'cov_bspline_te.txt')
    np.savetxt(cov_file_tr, X_tr)
    np.savetxt(cov_file_te, X_te)
    
    if not force_refit and os.path.exists(os.path.join(idp_dir, 'Models', 'NM_0_0_estimate.pkl')):
        print('Making predictions using a pre-existing model...')
        suffix = 'predict'

        # Make prdictsion with test data
        predict(cov_file_te,
                alg='blr',
                respfile=resp_file_te,
                model_path=os.path.join(idp_dir, 'Models'),
                outputsuffix=suffix)
    else:
        print('Estimating the normative model...')
        estimate(cov_file_tr, resp_file_tr, testresp=resp_file_te,
                 testcov=cov_file_te, alg='blr', optimizer='l-bfgs-b',  # optimizer='l-bfgs-b'
                 savemodel=True, warp=warp, warp_reparam=True)
        suffix = 'estimate'