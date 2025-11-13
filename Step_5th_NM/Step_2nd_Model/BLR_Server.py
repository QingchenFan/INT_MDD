import os
import numpy as np
import pandas as pd
from pcntoolkit.normative import estimate, predict
from pcntoolkit.util.utils import create_design_matrix


model_name = 'INT_246'

df_tr = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC_MDD/INT_NM/Feature/HC_Train_harmonized.csv')
df_te = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC_MDD/INT_NM/Feature/HC_DiscoverSet_harmonized.csv')

print(" -Data Done ! -")
#TODO:
processing_dir = "/Volumes/QC/INT/INT_BN246_HC_MDD/INT_NM/Results/INT246_BLR_20251111/NMResults/"  # 输出路径


brainRegion = df_tr.columns.tolist()
idp_ids = brainRegion[4:]
print(idp_ids)

# which data columns do we wish to use as covariates?
cols_cov = ['age', 'sex']#,'ICV_total_volume_cm3'

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
    print(idp_dir)

    os.makedirs(os.path.join(idp_dir), exist_ok=True)
    os.chdir(idp_dir)

    # extract the response variables for training and test set
    y_tr = df_tr[idp].to_numpy()  # 训练集脑区对应的数据

    y_te = df_te[idp].to_numpy()  # 测试集脑区对应的数据

    # write out the response variables for training and test
    resp_file_tr = os.path.join(idp_dir, 'resp_tr.txt')
    resp_file_te = os.path.join(idp_dir, 'resp_te.txt')
    np.savetxt(resp_file_tr, y_tr)
    np.savetxt(resp_file_te, y_te)

    # configure the design matrix
    X_tr = create_design_matrix(df_tr[cols_cov],
                                #site_ids=df_tr['sitenum'],
                                basis='bspline',
                                xmin=xmin,
                                xmax=xmax)
    X_te = create_design_matrix(df_te[cols_cov],
                                #site_ids=df_te['sitenum'],
                                #all_sites=site_ids,
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
        estimate(cov_file_tr,
                 resp_file_tr,
                 testresp=resp_file_te,
                 testcov=cov_file_te,
                 alg='blr',
                 optimizer='l-bfgs-b',  # optimizer='l-bfgs-b'
                 savemodel=True,
                 warp=warp,
                 cvfolds=10,
                 warp_reparam=True)
        suffix = 'estimate'