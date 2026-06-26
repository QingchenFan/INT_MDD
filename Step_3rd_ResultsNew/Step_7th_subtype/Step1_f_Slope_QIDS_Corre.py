import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import statsmodels.formula.api as smf

# 1. 读取数据（保留您的绝对路径）
data_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_INTSlope/Slope_QIDS/subtype2_slope_QIDS.csv'
df = pd.read_csv(data_path)

# =========================================================================
# 2. 控制协变量（age, sex, mean_fd），计算偏相关系数
# =========================================================================
# 通过回归分析计算回归残差（即剔除协变量影响后的纯净变异）
res_slope = smf.ols('slope ~ age + sex + mean_fd', data=df).fit().resid
res_hamd = smf.ols('QIDS ~ age + sex + mean_fd', data=df).fit().resid

# 计算皮尔逊偏相关系数 (Pearson Partial Correlation)
pearson_r, pearson_p = stats.pearsonr(res_hamd, res_slope)
print(f"Pearson Partial correlation: r = {pearson_r:.4f}, p = {pearson_p:.4f}")

# 计算斯皮尔曼偏相关系数 (Spearman Partial Correlation)
spearman_r, spearman_p = stats.spearmanr(res_hamd, res_slope)
print(f"Spearman Partial correlation: r = {spearman_r:.4f}, p = {spearman_p:.4f}")

# =========================================================================
# 3. 绘制偏回归散点图和趋势线
# =========================================================================
plt.figure(figsize=(6, 5))

# 绘制剔除协变量影响后的残差散点
plt.scatter(res_hamd, res_slope, alpha=0.7, color='gray', edgecolors='none')

# 计算并绘制残差的拟合线（趋势线）
m, b = np.polyfit(res_hamd, res_slope, 1)
x_vals = np.array([res_hamd.min(), res_hamd.max()])

# 修改图例：仅给出 r 值和 p 值，其他文字全部去除
legend_label = (
    f"Pearson r = {pearson_r:.3f}, p = {pearson_p:.3f}"
)
plt.plot(x_vals, m * x_vals + b, color='red', label=legend_label)

# 设置图表标签和标题
plt.xlabel('QIDS')
plt.ylabel('Slope ')
plt.title('Partial Correlation: QIDS vs Slope')
plt.legend(loc='upper right')
plt.tight_layout()

# 4. 保存图片（保留您的绝对路径）
output_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_INTSlope/Slope_QIDS/subtype2_slope_QIDS.png'
plt.savefig(output_path, dpi=300)
print(f"Plot saved to: {output_path}")