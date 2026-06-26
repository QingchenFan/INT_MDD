import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection
'''
    此代码计算 INT共变-考虑到了年龄 性别 FD ,用此代码计算 INT共变网络
'''
# 1. 加载数据
df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype2_INT_agesex_FD.csv')

# 2. 定义协变量和脑区列
covariates = ['age', 'sex', 'mean_fd']
# 前4列为 subID, mean_fd, age, sex，后面的 246 列为脑区 INT 值
regions = df.columns[4:]

# 3. 线性回归提取残差：剔除协变量影响
# 构建自变量矩阵 X (加上截距项)
X = df[covariates].values
X = sm.add_constant(X)

# 初始化残差矩阵 (受试者数量 x 246个脑区)
residuals = np.zeros((df.shape[0], len(regions)))

for i, region in enumerate(regions):
    y = df[region].values
    model = sm.OLS(y, X).fit()
    residuals[:, i] = model.resid

# 4. 计算结构共变网络 (残差的皮尔逊相关即为偏相关)
r_matrix = np.corrcoef(residuals, rowvar=False)

# 5. 计算相应的 P 值
n_samples = df.shape[0]  # 受试者人数 N=156
k_covariates = 3         # 协变量个数 k=3
df_err = n_samples - 2 - k_covariates # 自由度 df = N - 2 - k = 151

# 限制 r_matrix 在 (-0.999999, 0.999999) 避免除以0的报错
r_mat_clipped = np.clip(r_matrix, -0.999999, 0.999999)
# 计算 t 统计量
t_matrix = r_mat_clipped * np.sqrt(df_err / (1 - r_mat_clipped**2))
# 根据 t 分布计算双侧 P 值
p_matrix = stats.t.sf(np.abs(t_matrix), df_err) * 2

# 对角线是脑区自己和自己的相关（r=1），将 P 值设为 1
np.fill_diagonal(p_matrix, 1)

# 6. FDR 多重比较校正 (仅提取矩阵的右上半角进行校正)
upper_tri_indices = np.triu_indices_from(p_matrix, k=1)
p_values_upper = p_matrix[upper_tri_indices]

# 使用 Benjamini-Hochberg 方法进行 FDR 校正
rejected, p_values_fdr_upper = fdrcorrection(p_values_upper, alpha=0.05, method='indep')

# 将校正后的 P 值还原回对称矩阵
p_matrix_fdr = np.ones_like(p_matrix)
p_matrix_fdr[upper_tri_indices] = p_values_fdr_upper

# 镜像对称到左下半角
lower_tri_indices = np.tril_indices_from(p_matrix_fdr, k=-1)
p_matrix_fdr[lower_tri_indices] = p_matrix_fdr.T[lower_tri_indices]

# 7. 转换为 DataFrame 格式并保存为 CSV
r_df = pd.DataFrame(r_matrix, index=regions, columns=regions)
p_df = pd.DataFrame(p_matrix, index=regions, columns=regions)
fdr_df = pd.DataFrame(p_matrix_fdr, index=regions, columns=regions)

r_df.to_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationII/subtype2_INT_covariance_network_r.csv')
p_df.to_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationII/subtype2_INT_covariance_network_p.csv')
fdr_df.to_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationII/subtype2_INT_covariance_network_fdr.csv')

print("计算完成，文件已保存！")