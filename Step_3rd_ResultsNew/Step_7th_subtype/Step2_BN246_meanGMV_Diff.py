import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 读取数据与合并 (请检查本地路径)
# ==========================================
# 假设文件在当前运行目录下，否则请加上完整路径
df_s1 = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step3_cGMVmean_Diff/subtype1_GMV246.csv')
df_s2 = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step3_cGMVmean_Diff/subtype2_GMV246.csv')

# 打上组别标签
df_s1['Group'] = 'Subtype1'
df_s2['Group'] = 'Subtype2'

# 合并数据并提取需要的列（去掉含缺失值的行）
df = pd.concat([df_s1, df_s2], ignore_index=True)
df = df[['Group', 'age', 'sex', 'TIV', 'mean_GMV']].dropna()

# ==========================================
# 2. 广义线性模型 (GLM) / 协方差分析 (ANCOVA)
# ==========================================
# 将 sex 强制作为分类变量 C(sex) 纳入模型
formula = 'mean_GMV ~ C(Group) + age + C(sex) + TIV'
model = smf.ols(formula, data=df).fit()

# 计算 ANOVA 表，检验各变量的主效应
anova_table = sm.stats.anova_lm(model, typ=2)
print("=== 协方差分析 (ANCOVA) 表格 ===")
print(anova_table)

# 提取组间比较 (Subtype2 相比于 Subtype1) 的真实 T 值和 P 值
target_param = 'C(Group)[T.Subtype2]'
p_val = model.pvalues.get(target_param, np.nan)
t_val = model.tvalues.get(target_param, np.nan)

print("\n=== 控制年龄、性别、TIV后的组间真实差异 ===")
print(f"t-value = {t_val:.4f}")
print(f"p-value = {p_val:.6e}")

# ==========================================
# 3. 绘制并动态智能标注显著性的箱体图
# ==========================================
plt.figure(figsize=(6, 5))
sns.set(style="whitegrid")

# 绘制箱线图与底层散点图
ax = sns.boxplot(x='Group', y='mean_GMV', data=df, order=['Subtype1', 'Subtype2'], palette='Set3', width=0.5,
                 showfliers=False)
sns.stripplot(x='Group', y='mean_GMV', data=df, order=['Subtype1', 'Subtype2'], color='black', alpha=0.3, size=4,
              jitter=True)

# 计算星号连线的位置
y_max = df['mean_GMV'].max()
y_min = df['mean_GMV'].min()
y_range = y_max - y_min
h = y_range * 0.05
line_y = y_max + h

# 只有当模型校正后的 p 值显著时，才画出星号
if p_val < 0.05:
    x1, x2 = 0, 1
    # 绘制倒 U 型连线
    plt.plot([x1, x1, x2, x2], [line_y, line_y + h, line_y + h, line_y], lw=1.5, c='black')

    # 判断要标几颗星
    if p_val < 0.001:
        stars = "***"
    elif p_val < 0.01:
        stars = "**"
    else:
        stars = "*"

    plt.text((x1 + x2) * .5, line_y + h, stars, ha='center', va='bottom', color='black', fontsize=18, fontweight='bold')
    sig_text = f"p < {0.001 if p_val < 0.001 else round(p_val, 3)}"
else:
    sig_text = "ns (Not Significant)"

# 预留出连线和星号的垂直空间，防止被裁切
plt.ylim(y_min - y_range * 0.05, line_y + h * 3)

plt.title(f'Comparison of mean_GMV between Subtypes\n(Covariates: Age, Sex, TIV | {sig_text})', fontsize=13, pad=15)
plt.ylabel('Mean Gray Matter Volume (mean_GMV)', fontsize=12)
plt.xlabel('Group', fontsize=12)
plt.tight_layout()

# 保存高质量图片
out_fig = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step3_cGMVmean_Diff/mean_GMV_Subtype1_Subtype2_GLMagesex.png'
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
plt.show()

print(f"\n=== 绘图完成！图片已保存至: {out_fig} ===")