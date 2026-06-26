import glob
import pandas as pd
#path = '/Volumes/QCII/BrainProject/xcpd_out_PD/xcp_d/sub-*/func/*task-rest_acq-ap_run-1_outliers.tsv'
path = '/Volumes/QCII/duilie_processed/duilie_HC_MDD_fmriprep/sub-HC*V01/sub-HC*V01/func/sub-HC*V01_task-rest_acq-ap_run-1_desc-confounds_timeseries.tsv'
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
df.to_csv('./excluded_subjectsDL.csv', index=False)