import glob
import pandas as pd
#path = '/Volumes/QCII/BrainProject/xcpd_out_PD/xcp_d/sub-*/func/*task-rest_acq-ap_run-1_outliers.tsv'
path = '/Volumes/QCCC/HCP_xcpd_out2/sub-*/func/sub-*_task-REST1_dir-LR_outliers.tsv'
fl = glob.glob(path)
subbox = []
for i in fl:
    id = i.split('/')[-3]
    print(id)
    conf = pd.read_csv(i, sep='\t')
    fd = conf['framewise_displacement'][:]
    a = fd[fd >= 1]
    res = len(a) / len(fd)
    if res > 0.20:
        subbox.append(id)
        print('exclude:', id, '%:', res)

df = pd.DataFrame(subbox, columns=['subID'])
df.to_csv('./excluded_subjectsHCP2.csv', index=False)