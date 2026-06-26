import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg') # 确保在服务器环境下正常绘图
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

# 1. 数据读取与准备
path_base = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_INTSlope/'
df_hc = pd.read_csv(path_base + 'HC_INT_slopes.csv')
df1 = pd.read_csv(path_base + 'subtype1_INT_slopes.csv')
df2 = pd.read_csv(path_base + 'subtype2_INT_slopes.csv')

df_hc['Group'] = 'HC'
df1['Group'] = 'Subtype1'
df2['Group'] = 'Subtype2'
df_combined = pd.concat([df_hc, df1, df2], ignore_index=True)

# 2. 统计学比较 (线性模型，以 HC 为参考组)
model = smf.ols("slope ~ C(Group, Treatment(reference='HC')) + age + C(sex) + mean_fd",
                data=df_combined).fit()
print("=== 三组间统计比较结果 ===")
print(model.summary())

# 3. 绘图准备
sns.set_theme(style="ticks", context="talk")
plt.figure(figsize=(10, 6))

# 使用 hue 参数以符合 Seaborn 新版本规范
ax = sns.boxplot(x='Group', y='slope', hue='Group', data=df_combined,
                 width=0.5, palette="Set2", showfliers=False,
                 boxprops=dict(alpha=0.8), legend=False)
sns.stripplot(x='Group', y='slope', hue='Group', data=df_combined,
              size=4, color="black", alpha=0.4, jitter=0.2, legend=False)

# 4. 自动标注显著性 (包含所有两两比较)
pairs = [("HC", "Subtype1"), ("HC", "Subtype2"), ("Subtype1", "Subtype2")]

# 获取 p 值并强制转换为 float (修复 ValueError 报错的关键)
p_hc_s1 = float(model.pvalues["C(Group, Treatment(reference='HC'))[T.Subtype1]"])
p_hc_s2 = float(model.pvalues["C(Group, Treatment(reference='HC'))[T.Subtype2]"])

# 计算 S1 vs S2 的差异
t_test_res = model.t_test("C(Group, Treatment(reference='HC'))[T.Subtype1] - C(Group, Treatment(reference='HC'))[T.Subtype2] = 0")
p_s1_s2 = float(t_test_res.pvalue)

# 创建标注对象
annotator = Annotator(ax, pairs, x='Group', y='slope', data=df_combined)
annotator.configure(text_format="star", loc='inside', verbose=False)
# 传入三个对应的 float p 值
annotator.set_pvalues(pvalues=[p_hc_s1, p_hc_s2, p_s1_s2])
annotator.annotate()

# 5. 美化与保存
plt.title('Comparison of INT Slopes across Groups', fontsize=18, pad=20)
plt.ylabel('Slope', fontsize=14)
plt.xlabel('Group', fontsize=14)
sns.despine(trim=True, offset=10)
plt.tight_layout()

# 保存路径
output_path = path_base + 'three_group_slope_comparison_final.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图表已保存至: {output_path}")
plt.show()