import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from sklearn.linear_model import LinearRegression
import time

print("=== 阶段 1: 加载数据并计算真实的拓扑属性 ===")
# ------------------------------------------
# 1.1 加载 R 矩阵和 FDR 矩阵
# ------------------------------------------
r1_df = pd.read_csv(
    '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationII/subtype1_INT_covariance_network_r.csv',
    index_col=0)
p1_fdr_df = pd.read_csv(
    '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationII/subtype1_INT_covariance_network_fdr.csv',
    index_col=0)

r2_df = pd.read_csv(
    '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationII/subtype2_INT_covariance_network_r.csv',
    index_col=0)
p2_fdr_df = pd.read_csv(
    '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationII/subtype2_INT_covariance_network_fdr.csv',
    index_col=0)

regions = r1_df.columns

# ------------------------------------------
# 1.2 您的原函数：计算单组基于 FDR 阈值的节点强度
# ------------------------------------------
def calculate_node_strength_true(r_matrix, fdr_matrix):
    r_clipped = np.clip(r_matrix.values, -0.999999, 0.999999)
    z_matrix = np.arctanh(r_clipped)
    mask = (fdr_matrix.values < 0.05).astype(int)
    z_significant = z_matrix * mask
    node_strength = np.abs(z_significant).sum(axis=1)
    degree = mask.sum(axis=1)
    return node_strength, degree


strength1, degree1 = calculate_node_strength_true(r1_df, p1_fdr_df)
strength2, degree2 = calculate_node_strength_true(r2_df, p2_fdr_df)
true_diff = strength1 - strength2

print("\n=== 阶段 2: 加载原始数据并剔除协变量 (为置换检验做准备) ===")
# ------------------------------------------
# 2.1 加载含有协变量的受试者数据
# ------------------------------------------
df_sub1 = pd.read_csv(
    '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype1_INT_agesex_FD.csv')
df_sub2 = pd.read_csv(
    '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype2_INT_agesex_FD.csv')

n1, n2 = len(df_sub1), len(df_sub2)
covariates = ['subID', 'mean_fd', 'age', 'sex']
# 确保列名一致
regions_raw = [col for col in df_sub1.columns if col not in covariates]

df_all = pd.concat([df_sub1, df_sub2], ignore_index=True)
X_cov = df_all[['mean_fd', 'age', 'sex']].values
Y_feat = df_all[regions_raw].values

# 2.2 回归剔除协变量获取残差
model = LinearRegression().fit(X_cov, Y_feat)
Y_res = Y_feat - model.predict(X_cov)

print("\n=== 阶段 3: 执行严格的置换检验 (包含动态 FDR 校正) ===")

# ------------------------------------------
# 3.1 定义置换专用的网络重构函数
# 每次洗牌后，严格复现计算 r -> 算 p -> 算 FDR -> 掩膜过滤 -> 算强度
# ------------------------------------------
def calc_perm_strength_fdr(data, k_covariates=3):
    n = data.shape[0]
    r = np.corrcoef(data.T)
    np.fill_diagonal(r, 0)

    r_clip = np.clip(r, -0.9999, 0.9999)
    # 计算偏相关的 T 统计量，自由度为 n - 2 - k (k为协变量个数)
    df = n - 2 - k_covariates
    t = r_clip * np.sqrt(df / (1 - r_clip ** 2))
    p = 2 * stats.t.sf(np.abs(t), df=df)
    np.fill_diagonal(p, 1)

    # 提取上三角进行 FDR 校正，极大提升运算速度
    upper_idx = np.triu_indices_from(p, k=1)
    _, p_fdr_upper = fdrcorrection(p[upper_idx], alpha=0.05, method='indep')

    # 重构对称的 FDR 矩阵
    p_fdr = np.ones_like(p)
    p_fdr[upper_idx] = p_fdr_upper
    p_fdr.T[upper_idx] = p_fdr_upper

    # 过滤并计算节点强度
    mask = (p_fdr < 0.05).astype(int)
    z = np.arctanh(r_clip)
    strength = np.abs(z * mask).sum(axis=1)
    return strength

# ------------------------------------------
# 3.2 运行 1000 次置换
# ------------------------------------------
n_perm = 1000
fake_diffs = np.zeros((n_perm, len(regions)))
np.random.seed(42)

start_time = time.time()
print(f"开始 {n_perm} 次洗牌，每次洗牌将重新计算近 3 万条连线的 FDR，请耐心等待 1-2 分钟...")

for i in range(n_perm):
    # 打乱所有人
    idx = np.random.permutation(n1 + n2)
    fake_data1 = Y_res[idx[:n1], :]
    fake_data2 = Y_res[idx[n1:], :]

    # 计算假网络的 FDR 过滤强度
    fake_s1 = calc_perm_strength_fdr(fake_data1)
    fake_s2 = calc_perm_strength_fdr(fake_data2)

    # 记录假差异
    fake_diffs[i, :] = fake_s1 - fake_s2

    if (i + 1) % 100 == 0:
        elapsed = time.time() - start_time
        print(f"  已完成 {i + 1} 次置换... 耗时: {elapsed:.1f} 秒")


print("\n=== 阶段 4: 汇总生成终极拓扑比较结果 (包含 FDR 多重比较校正) ===")
# ------------------------------------------
# 4.1 计算双侧非参数 P 值
# ------------------------------------------
p_values_raw = np.mean(np.abs(fake_diffs) >= np.abs(true_diff), axis=0)

# ------------------------------------------
# 4.2 【新增】对 246 个脑区的置换 P 值进行 FDR 校正
# ------------------------------------------
rejected, p_values_fdr = fdrcorrection(p_values_raw, alpha=0.05, method='indep')

# ------------------------------------------
# 4.3 构建包含所有维度的最终数据框
# ------------------------------------------
topology_final_df = pd.DataFrame({
    'Region': regions,
    'Strength_Subtype1': strength1,
    'Degree_Subtype1': degree1,
    'Strength_Subtype2': strength2,
    'Degree_Subtype2': degree2,
    'Strength_Diff(S1-S2)': true_diff,
    'Abs_Strength_Diff': np.abs(true_diff),
    'P_value_Raw': p_values_raw,             # 原始置换 P 值
    'P_value_FDR': p_values_fdr,             # FDR 校正后的 P 值
    'Is_Significant_FDR': rejected           # 是否通过 FDR 校正 (True/False)
})

# 按原始 P 值从小到大排序，P值相同时按绝对强度差异降序排列
topology_final_df = topology_final_df.sort_values(by=['P_value_Raw', 'Abs_Strength_Diff'],
                                                  ascending=[True, False])

# 删除辅助排序用的绝对值列，保持表格整洁
topology_final_df = topology_final_df.drop(columns=['Abs_Strength_Diff'])

# 保存结果
save_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationII/result2_StrengthDiff/Final_Topology_Permutation_Comparison.csv'
topology_final_df.to_csv(save_path, index=False)

print("\n🎉 分析全部完成！数据已保存至：", save_path)
print("根据置换检验原始 P 值排名前 10 的核心改变脑区如下：")
print(topology_final_df.head(10).to_string(index=False))