import glob
import os.path

import pandas as pd
#path = '/Volumes/QCII/BrainProject/xcpd_out_PD/xcp_d/sub-*/func/*task-rest_acq-ap_run-1_outliers.tsv'
path = '/Volumes/QC/Data/INT/BN246timeseries_surface/MDD/sub-*V01'
fl = glob.glob(path)

subbox = []
for i in fl:
    subID = i.split('/')[-1]
    print(subID)

    confound = '/Volumes/ZLabData/BrainProject/brainproject_I/fmriprep_out_PD/'+subID+'/func/'+subID+'_task-rest_acq-ap_run-1_desc-confounds_timeseries.tsv'
    if not os.path.isfile(confound):
        continue
    conf = pd.read_csv(confound, sep='\t')
    fd = conf['framewise_displacement'][:]

    fd.iloc[0] = 0
    save_path = f'/Volumes/QC/Data/INT/BN246timeseries_surface/MDD/{subID}/{subID}_FD.txt'
    fd.to_csv(save_path, index=False, header=False)






