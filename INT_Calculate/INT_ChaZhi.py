import glob
import pandas as pd
import numpy as np
from statsmodels.tsa import stattools


# ============================================================
# 1. 参数设置
# ============================================================

TR = 2.0               # TR，单位：秒
nlags = 20             # 最大 lag
N_INTERP_POINTS = 50   # 插值后的点数


# ============================================================
# 2. 计算 ACF
# ============================================================
def acf_curve(icas, nlags):
    return np.array([
        stattools.acf(ica, nlags=nlags)
        for ica in icas
    ])


# ============================================================
# 3. 计算插值后的 INT
# ============================================================
def intrinsic_timescale(acf_data, delta_lag, TR=2.0):
    """
    排除插值序列第0点，累加首次负值之前的 ACF。
    乘 delta_lag 修正插值步长，再乘 TR 换算为秒。

    未出现负值时，返回最大 lag 窗口内的截断估计。
    """
    acf_data = np.asarray(acf_data, dtype=float)

    if not np.isfinite(acf_data).all():
        return np.nan

    negative_indices = np.flatnonzero(acf_data < 0)

    if negative_indices.size:
        first_neg = negative_indices[0]
        int_value = np.sum(acf_data[1:first_neg])
    else:
        int_value = np.sum(acf_data[1:])

    return TR * delta_lag * int_value


# ============================================================
# 4. 数据路径
# ============================================================
datapath = (
    "/Volumes/QC/Data/BN246timeseries_surface/"
    "MDD/BP_135/sub-*/*.txt"
)

data = sorted(glob.glob(datapath))
results = []


# ============================================================
# 5. 逐被试计算
# ============================================================
for file in data:

    if 'FD' in file:
        continue

    print(file)
    subID = file.split('/')[-2]

    # 输入格式：时间点 × 脑区
    timescale = np.loadtxt(file)

    # 计算各脑区原始 ACF
    timescale_acf_curve = acf_curve(timescale.T, nlags)

    int_subj = []

    for acf_data in timescale_acf_curve:

        # 对 ACF 进行线性插值
        x = np.arange(len(acf_data))
        xvals = np.linspace(x[0], x[-1], N_INTERP_POINTS)
        yinterp = np.interp(xvals, x, acf_data)
        # 插值后的 lag 步长
        delta_lag = xvals[1] - xvals[0]
        # 计算秒尺度的 INT 面积近似
        int_value = intrinsic_timescale(
            yinterp,
            delta_lag=delta_lag,
            TR=TR,
        )
        int_subj.append(int_value)
    results.append([subID] + int_subj)


# ============================================================
# 6. 保存 CSV
# ============================================================

df = pd.DataFrame(results)

output_file = "./MDD_INT_interpolation.csv"
df.to_csv(output_file, index=False, encoding="utf-8")

print("\n========================================")
print("INT calculation finished.")
print("TR:", TR, "seconds")
print("nlags:", nlags)
print("Interpolation points:", N_INTERP_POINTS)
print("Output:", output_file)
print("========================================")