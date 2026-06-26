import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.conftest import matplotlib

# 若不需要在本地弹窗，保留 Agg
matplotlib.use('Agg')

# ==========================================
# 1. 读取数据
# ==========================================
base_dir = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_INTRange/Step1_BN246_INTRange/'
df1 = pd.read_csv(base_dir + 'subtype1_BN246INT_Range.csv')
df2 = pd.read_csv(base_dir + 'subtype2_BN246INT_Range.csv')

# ==========================================
# 2. 添加组别标签并合并数据
# ==========================================
df1['group'] = 'Subtype1'
df2['group'] = 'Subtype2'

# 将两组数据合并，并删除含有缺失值的行
df = pd.concat([df1, df2], ignore_index=True)
df = df.dropna(subset=['difference', 'age', 'sex', 'mean_fd'])

# ==========================================
# 3. 统计检验：协方差分析 (ANCOVA)
# ==========================================
# 公式：difference 受到 group, age, sex, mean_fd 的共同影响
model = ols('difference ~ C(group) + age + C(sex) + mean_fd', data=df).fit()

# 计算 ANOVA 表 (使用 Type 2 误差平方和)
anova_table = sm.stats.anova_lm(model, typ=2)

# 提取组别差异的 P 值和 F 值
p_val = anova_table.loc['C(group)', 'PR(>F)']
f_val = anova_table.loc['C(group)', 'F']

print("========== ANCOVA 统计结果 ==========")
print(anova_table)

# ==========================================
# 4. 动态星号标记函数
# ==========================================
def get_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'

star_text = get_stars(p_val)

# ==========================================
# 5. 绘图对比与显著性标注
# ==========================================
plt.figure(figsize=(8, 6))

# 绘制箱体图
sns.boxplot(x='group', y='difference', data=df, order=['Subtype1', 'Subtype2'], palette='Set2', showfliers=True)
# 叠加散点图，直观展示数据分布
sns.stripplot(x='group', y='difference', data=df, order=['Subtype1', 'Subtype2'], color='black', alpha=0.4, jitter=True)

# ----- 开始画显著性标注 (连线和星号) -----
# 获取当前数据的最大最小值，用来按比例计算连线的高度
y_max = df['difference'].max()
y_min = df['difference'].min()
y_range = y_max - y_min

# 设定线段的高度和向下的"小尾巴"高度
line_y = y_max + 0.05 * y_range
h = 0.02 * y_range

# 画横线及两端的竖线：从 x=0(Subtype1) 到 x=1(Subtype2)
plt.plot([0, 0, 1, 1], [line_y, line_y+h, line_y+h, line_y], lw=1.5, color='black')
# 在线段中心偏上方添加星号文字
plt.text(0.5, line_y+h + 0.01*y_range, star_text, ha='center', va='bottom', color='black', fontsize=14, fontweight='bold')

# 自动调整 y 轴的高度限制，防止顶部的星号被裁剪掉
plt.ylim(y_min - 0.05 * y_range, line_y + 0.15 * y_range)
# ----------------------------------------

# 动态生成包含 P 值的标题
title_text = f'Comparison of INT Difference (Max - Min)\nANCOVA (controlling age, sex, mean_fd)\n$p={p_val:.4e}$ (F={f_val:.2f})'
plt.title(title_text, fontsize=14, pad=15)
plt.ylabel('Difference (Max - Min)', fontsize=12)
plt.xlabel('Subtype', fontsize=12)

plt.tight_layout()

# 保存并展示图片
save_path = base_dir + 'subtype1_subtype2_range_difference_Stars.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"\n绘图完成！已保存至 {save_path}")