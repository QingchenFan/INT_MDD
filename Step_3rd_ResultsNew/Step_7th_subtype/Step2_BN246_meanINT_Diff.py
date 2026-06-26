import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 读取数据与合并
# ==========================================
# ⚠️ 注意：请确保您读取的这两个 csv 文件中确实包含 'mean_fd' 列。
# 如果不包含，请将路径替换为您之前带FD后缀的文件，例如 'subtype1_INT_agesex_FD.csv'
df_s1 = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step2_BN246mean_INT/subtype1_mean_INT_agesex_FD.csv')
df_s2 = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step2_BN246mean_INT/subtype2_mean_INT_agesex_FD.csv')

df_s1['Group'] = 'Subtype1'
df_s2['Group'] = 'Subtype2'

# 合并两组数据
df = pd.concat([df_s1, df_s2], ignore_index=True)

# 【修改点1】：提取所需列时，加入 'mean_fd' 并去除缺失值
df = df[['Group', 'age', 'sex', 'mean_INT', 'mean_fd']].dropna()

# ==========================================
# 2. 广义线性模型 (GLM) / 协方差分析 (ANCOVA)
# ==========================================
# 【修改点2】：将 mean_fd 放入回归公式中
formula = 'mean_INT ~ C(Group) + age + C(sex) + mean_fd'
model = ols(formula, data=df).fit()

# 计算 ANOVA 表，检验组间主效应
anova_table = sm.stats.anova_lm(model, typ=2)
print("=== 协方差分析 (ANCOVA) 结果 ===")
print(anova_table)

# 提取 P 值和回归系数
target_param = "C(Group)[T.Subtype2]"
p_val = model.pvalues.get(target_param, np.nan)
t_val = model.tvalues.get(target_param, np.nan)
print(f"\n控制年龄、性别和头动(mean_fd)后，组间差异的真实 P 值: {p_val:.6e}")

# ==========================================
# 3. 绘制并动态智能标注显著性的箱体图
# ==========================================
plt.figure(figsize=(6, 5))
sns.set(style="whitegrid")

# 绘制箱体图和散点图
ax = sns.boxplot(x='Group', y='mean_INT', data=df, order=['Subtype1', 'Subtype2'], palette='Set2', width=0.5,
                 showfliers=False)
sns.stripplot(x='Group', y='mean_INT', data=df, order=['Subtype1', 'Subtype2'], color='black', alpha=0.3, size=4,
              jitter=True)

# 计算画线位置
y_max = df['mean_INT'].max()
y_range = df['mean_INT'].max() - df['mean_INT'].min()
h = y_range * 0.05
line_y = y_max + h

# 如果 p_val < 0.05，则绘制连线和星号
if p_val < 0.05:
    x1, x2 = 0, 1
    # 绘制倒 U 型连线
    plt.plot([x1, x1, x2, x2], [line_y, line_y + h, line_y + h, line_y], lw=1.5, c='black')

    # 判断星号数量
    if p_val < 0.001:
        stars = "***"
    elif p_val < 0.01:
        stars = "**"
    else:
        stars = "*"

    plt.text((x1 + x2) * .5, line_y + h, stars, ha='center', va='bottom', color='black', fontsize=16, fontweight='bold')

# 留出顶部空间
plt.ylim(df['mean_INT'].min() - y_range * 0.05, line_y + h * 3)

# 【修改点3】：更新了图表标题，展示出也控制了 mean_fd
plt.title('Comparison of mean_INT between Subtypes\n(Covariates: Age, Sex, mean_fd)', fontsize=13, pad=15)
plt.ylabel('Mean INT Value', fontsize=12)
plt.xlabel('Group', fontsize=12)
plt.tight_layout()

# 【修改点4】：保存图片的文件名后加上了 FD，避免覆盖之前的图像
out_fig = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step2_BN246mean_INT/mean_INT_Subtype1_vs_Subtype2_GLMagesexFD.png'
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
#plt.show()

print(f"\n=== 绘图完成！图片已保存至: {out_fig} ===")