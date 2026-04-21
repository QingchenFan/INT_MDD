import glob
import scipy.io as sio
import pandas as pd
import numpy as np


path = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZII/INT_value/INT_HCind/*"
file_list = glob.glob(path)

all_data = []

for mat_path in file_list:
    # 提取 subID
    subID = mat_path.split('/')[-1].split('_')[0]

    # 加载 mat 文件
    mat_data = sio.loadmat(mat_path)['subj_hwhm']

    mat_data = sio.loadmat(mat_path)
    values = mat_data['subj_hwhm']

    values_flat = values.ravel()

    # 拼接：subID + 数据
    row = [subID] + values_flat.tolist()
    all_data.append(row)

# 保存为 CSV
df = pd.DataFrame(all_data)
df.to_csv("./HC_INT2.csv", index=False, header=False)


