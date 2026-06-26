import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection


def calculate_gmv_covariance_network(input_path, output_prefix):
    print(f"正在处理数据: {input_path} ...")

    # 1. 加载数据
    df = pd.read_csv(input_path)

    # 2. 定义协变量和脑区列
    # GMV 的协变量：年龄、性别、TIV(颅内总体积)
    covariates = ['age', 'sex', 'TIV']

    # 前4列为 subID, TIV, age, sex，后面的 246 列为脑区 GMV 值
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
    n_samples = df.shape[0]  # 受试者人数 N
    k_covariates = len(covariates)  # 协变量个数 k=3
    df_err = n_samples - 2 - k_covariates  # 自由度 df = N - 2 - k

    # 限制 r_matrix 在 (-0.999999, 0.999999) 避免除以0的报错
    r_mat_clipped = np.clip(r_matrix, -0.999999, 0.999999)
    # 计算 t 统计量
    t_matrix = r_mat_clipped * np.sqrt(df_err / (1 - r_mat_clipped ** 2))
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

    # 保存文件 (根据前缀自动生成文件名)
    r_df.to_csv(f'{output_prefix}_GMV_covariance_network_r.csv')
    p_df.to_csv(f'{output_prefix}_GMV_covariance_network_p.csv')
    fdr_df.to_csv(f'{output_prefix}_GMV_covariance_network_fdr.csv')

    print(f"[{output_prefix}] 计算完成，文件已保存！\n")


# ==============================================================
# 批量处理 Subtype 1 和 Subtype 2
# 请替换为您本地实际的文件夹路径，以确保文件能正确找到和保存
# ==============================================================

# 定义输出的基础文件夹路径
out_dir = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationGMV'

# 处理 Subtype 1
calculate_gmv_covariance_network(
    input_path='/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype1_GMV246.csv',
    output_prefix=f'{out_dir}/subtype1'
)

# 处理 Subtype 2
calculate_gmv_covariance_network(
    input_path='/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype2_GMV246.csv',
    output_prefix=f'{out_dir}/subtype2'
)

print("所有 GMV 共变网络计算完毕！")