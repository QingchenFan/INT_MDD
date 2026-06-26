import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.conftest import matplotlib

matplotlib.use('Agg')
# 1. 读取两组数据（请将此处替换为你的实际文件路径）
df1 = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_INTSlope/subtype1_INT_slopes.csv')
df2 = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_INTSlope/subtype2_INT_slopes.csv')

# 2. 添加组别标签，合并数据
df1['Group'] = 'Subtype1'
df2['Group'] = 'Subtype2'
df_combined = pd.concat([df1, df2], ignore_index=True)

# 3. 统计学比较 (控制协变量的 GLM / ANCOVA)
# 使用普通的 slope (不做 zscore) 拟合模型
model = smf.ols("slope ~ C(Group) + age + C(sex) + mean_fd", data=df_combined).fit()
print("=== 组间统计比较结果 ===")
print(model.summary())

# 4. 绘制带有散点的箱体图 (发表级科研插图)
sns.set_theme(style="ticks", context="talk")
plt.figure(figsize=(8, 6))

# a. 绘制箱体图 (隐藏离群点以避免与后续的散点重复)
ax = sns.boxplot(x='Group', y='slope', data=df_combined,
                 width=0.5, palette="Set2", showfliers=False,
                 boxprops=dict(alpha=0.8))

# b. 叠加个体散点图
sns.stripplot(x='Group', y='slope', data=df_combined,
              size=5, color="black", alpha=0.5, jitter=0.2)

# c. 图表美化与标签设置
plt.title('Comparison of INT Slopes', fontsize=18, pad=20)
plt.ylabel('Slope', fontsize=14)
plt.xlabel('Subtype Group', fontsize=14)
sns.despine(trim=True, offset=10) # 移除多余的边框

# d. 动态添加显著性星号 (根据我们算出的 p < 0.001)
y_max = df_combined['slope'].max()
y_range = df_combined['slope'].max() - df_combined['slope'].min()
y_offset = y_range * 0.05

# 绘制表示组间比较的横线和垂直短线
plt.plot([0, 0, 1, 1],
         [y_max + y_offset, y_max + y_offset*1.5, y_max + y_offset*1.5, y_max + y_offset],
         lw=1.5, c='black')

# 写上星号
plt.text(0.5, y_max + y_offset*1.6, "***", ha='center', va='bottom', color='black', fontsize=20)

# 5. 显示并保存高分辨率图片
plt.tight_layout()
plt.show()

# 如果需要保存到本地，请取消注释并指定路径
plt.savefig('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_INTSlope/'
            'subtype1_subtype2_slope_comparison.png', dpi=300, bbox_inches='tight')