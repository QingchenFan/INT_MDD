import glob
import pandas as pd
import numpy as np
from statsmodels.tsa import stattools
def acf_curve(icas,nlags):
  return np.array([stattools.acf(ica, nlags=nlags) for ica in icas])

# def intrinic_timescale_interp(intter):
#     # autocorr = stattools.acf(ica)
#     return np.sum(intter[1:list(intter<0).index(True)])
def intrinic_timescale_interp(intter):
    """
    计算 Intrinsic Timescale
    如果 ACF 一直为正，则累加所有正的部分（或做适当衰减）
    """
    try:
        # 正常情况：找到第一个负值的位置
        first_neg = list(intter < 0).index(True)
        return np.sum(intter[1:first_neg])

    except ValueError:
        # 异常情况：ACF 全为正数
        # 方案1（推荐）：直接累加所有 lag（从1开始）
        return np.sum(intter[1:])

datapath = "/Volumes/QC/Data/BN246timeseries_surface/MDD/DZ/sub-*/*.txt"
data = glob.glob(datapath)
results = []
nlags = 20
for i in data:
    if 'FD' in i:
        continue
    print(i)
    subID = i.split('/')[-2]

    timescale = np.loadtxt(i)

    all_auc_acf_interp = []
    timescale_acf_curve = acf_curve(timescale.T,nlags)

    for acf_curve_data in timescale_acf_curve:

        x = np.arange(len(acf_curve_data))

        xvals = np.linspace(0, nlags, 50)

        yinterp = np.interp(xvals, x, acf_curve_data)
        all_auc_acf_interp.append(yinterp)

    all_auc_acf_interp = np.array(all_auc_acf_interp)

    int_subj_interp = np.array([intrinic_timescale_interp(interpp) for interpp in all_auc_acf_interp])

    results.append([subID] + int_subj_interp.tolist())

# 构建DataFrame并保存
df = pd.DataFrame(results)
df.to_csv("./MDD_DZ_INT.csv", index=False, encoding="utf-8")