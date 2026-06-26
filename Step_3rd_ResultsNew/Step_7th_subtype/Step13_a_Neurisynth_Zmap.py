import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
# === 新增：导入多重比较校正模块 ===
from statsmodels.stats.multitest import fdrcorrection

# 1. 读取数据
df_sub = pd.read_csv("/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype2_INT_agesex.csv")
df_hc = pd.read_csv("/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step2_Ttest/HC_INT20_agesex.csv")

# 2. 添加组别标签 (例如：疾病亚型组为1，健康对照组为0)
df_sub['Group'] = 1
df_hc['Group'] = 0

# 3. 合并两个数据集
df = pd.concat([df_sub, df_hc], ignore_index=True)

# 4. 识别脑区列名 (排除非脑区特征列)
exclude_cols = ['subID', 'age', 'sex', 'Group']
regions = [col for col in df.columns if col not in exclude_cols]

# 确保 age 和 sex 是数值类型（防止格式读取错误）
df['age'] = pd.to_numeric(df['age'])
df['sex'] = pd.to_numeric(df['sex'])

# 5. 准备自变量矩阵 (X)
X = df[['Group', 'age', 'sex']]
X = sm.add_constant(X)

# 6. 循环遍历每个脑区，进行回归分析
results = []

for region in regions:
    y = df[region]

    # 拟合 OLS (普通最小二乘) 模型
    model = sm.OLS(y, X).fit()

    # 提取'Group' (组别主效应) 的 T值和 P值
    t_val = model.tvalues['Group']
    p_val = model.pvalues['Group']

    # 计算 Z 值 (双尾转换)
    p_val_clipped = max(p_val, 1e-300)
    z_val = norm.isf(p_val_clipped / 2) * np.sign(t_val)

    # 将结果保存到字典
    results.append({
        'BrainRegion': region,
        't_value': t_val,
        'z_value': z_val,
        'p_value': p_val
    })

# 7. 转换为 DataFrame
results_df = pd.DataFrame(results)
results_df.columns = ['脑区名称', 't值', 'z值', 'p值']

# === 新增：对所有的 p 值进行 FDR 校正 ===
# fdrcorrection 函数会返回两个数组：第一个是是否拒绝原假设的布尔值，第二个是校正后的 p 值。
# 我们只需要提取第二个数组放入新列即可。
rejected, fdr_p_values = fdrcorrection(results_df['p值'], alpha=0.05, method='indep')
results_df['FDR_p值'] = fdr_p_values
# =======================================

# 8. 保存结果
output_filename = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step15_Neurosynth/Neuronsynth_II/S2_vs_HC_zmap_INT.csv'
results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"计算完成！结果已保存至 {output_filename}")
