import glob

import numpy as np
import os
# 将 xcpd 输出的 tsv文件直接保存成 txt
path = '/Volumes/QC/Data/Shaefer400/MDD/sub-*/sub-*_task-rest_acq-ap_run-1_space-fsLR_atlas-4S456Parcels_timeseries.tsv'

box = glob.glob(path)
for i in box:
    subID = i.split('/')[-2]
    print(subID)

    matrix = np.genfromtxt(i, delimiter='\t', skip_header=1)


    newpath = "/Volumes/QC/Data/Shaefer400/MDD/" + subID
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    newdatap = newpath + '/' + subID + '.txt'

    np.savetxt(newdatap, matrix)

