import glob
import pandas as pd
#path = '/Volumes/QCII/BrainProject/xcpd_out_PD/xcp_d/sub-*/func/*task-rest_acq-ap_run-1_outliers.tsv'
path = '/Volumes/QC/Data/BN246timeseries_surface/HC/BP_III/sub-*V01/sub-*V01_task-rest_acq-ap_run-1_desc-confounds_timeseries.tsv'
fl = glob.glob(path)
subbox = []
for i in fl:
    id = i.split('/')[-3]
    conf = pd.read_csv(i, sep='\t')
    fd = conf['framewise_displacement'][:]
    a = fd[fd >= 1]
    res = len(a) / len(fd)
    if res > 0.20:
        subbox.append(id)
        print('exclude:', id, '%:', res)

df = pd.DataFrame(subbox, columns=['subID'])
df.to_csv('./excluded_subjectsBPIII_HC.csv', index=False)