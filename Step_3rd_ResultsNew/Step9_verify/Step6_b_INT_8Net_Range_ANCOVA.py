import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 若不需要在本地直接弹窗，可以使用 Agg 后端
matplotlib.use('Agg')

# ==========================================
# 1. 读取数据
# ==========================================
# 建议修改为您实际的本地 base_dir 路径
base_dir = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_verify/Step5_Range_Diff/'
df_hc = pd.read_csv(base_dir + 'HC_DZ_INT_8net_Range.csv')
df1 = pd.read_csv(base_dir + 'subtype1_DZ_INT_8net_Range.csv')
df2 = pd.read_csv(base_dir + 'subtype2_DZ_INT_8net_Range.csv')

# ==========================================
# 2. 添加组别标签并合并数据
# ==========================================
df_hc['group'] = 'HC'
df1['group'] = 'Subtype1'
df2['group'] = 'Subtype2'

# 将三组数据合并，并删除含有缺失值的行
df = pd.concat([df_hc, df1, df2], ignore_index=True)
df = df.dropna(subset=['difference', 'age', 'sex', 'mean_fd'])

# 为了在模型中有明确的对比基准，我们将 group 转换为有顺序的分类变量
df['group'] = pd.Categorical(df['group'], categories=['HC', 'Subtype1', 'Subtype2'])

# ==========================================
# 3. 统计检验：协方差分析 (ANCOVA) 主效应
# ==========================================
# 公式：difference 受到 group, age, sex, mean_fd 的共同影响
# 使用 Treatment("HC") 明确将 HC 作为基线对照组
model = ols('difference ~ C(group, Treatment("HC")) + age + C(sex) + mean_fd', data=df).fit()

# 计算 ANOVA 表 (使用 Type 2 误差平方和)
anova_table = sm.stats.anova_lm(model, typ=2)
print("========== 1. ANCOVA 主效应统计结果 ==========")
print(anova_table)

# ==========================================
# 4. 事后检验 (Pairwise Comparisons) 与 FDR 校正
# ==========================================
pairwise_results = []
contrasts = {
    'HC_vs_Subtype1': 'C(group, Treatment("HC"))[T.Subtype1] = 0',
    'HC_vs_Subtype2': 'C(group, Treatment("HC"))[T.Subtype2] = 0',
    'Subtype1_vs_Subtype2': 'C(group, Treatment("HC"))[T.Subtype1] - C(group, Treatment("HC"))[T.Subtype2] = 0'
}

# 计算每一对的差异 t 检验
for comp_name, contrast_str in contrasts.items():
    t_test_res = model.t_test(contrast_str)
    p_val = t_test_res.pvalue.item() if hasattr(t_test_res.pvalue, 'item') else float(t_test_res.pvalue)
    t_val = t_test_res.tvalue.item() if hasattr(t_test_res.tvalue, 'item') else float(t_test_res.tvalue)
    pairwise_results.append({
        'Comparison': comp_name,
        't_value': t_val,
        'p_uncorrected': p_val
    })

pairwise_df = pd.DataFrame(pairwise_results)

# 使用 FDR (Benjamini-Hochberg) 方法进行多重比较校正
_, p_fdr, _, _ = multipletests(pairwise_df['p_uncorrected'], alpha=0.05, method='fdr_bh')
pairwise_df['p_FDR'] = p_fdr

# 动态生成星号标注
def get_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'

pairwise_df['stars'] = pairwise_df['p_FDR'].apply(get_stars)

print("\n========== 2. 事后两两比较 (FDR 校正) ==========")
print(pairwise_df)

# ==========================================
# 5. 绘图对比与显著性标注
# ==========================================
plt.figure(figsize=(9, 6))

# 绘制箱体图
sns.boxplot(x='group', y='difference', data=df, order=['HC', 'Subtype1', 'Subtype2'], palette='Set2', showfliers=True)
# 叠加散点图
sns.stripplot(x='group', y='difference', data=df, order=['HC', 'Subtype1', 'Subtype2'], color='black', alpha=0.3, jitter=True)

# ----- 自动标注显著性连线和星号 -----
y_max = df['difference'].max()
y_min = df['difference'].min()
y_range = y_max - y_min

step = 0.08 * y_range
h = 0.02 * y_range
current_y = y_max + 0.05 * y_range

star_hc_s1 = pairwise_df[pairwise_df['Comparison'] == 'HC_vs_Subtype1']['stars'].values[0]
star_hc_s2 = pairwise_df[pairwise_df['Comparison'] == 'HC_vs_Subtype2']['stars'].values[0]
star_s1_s2 = pairwise_df[pairwise_df['Comparison'] == 'Subtype1_vs_Subtype2']['stars'].values[0]

def add_bracket(x1, x2, y, text_star):
    # 如果是不显著 (ns)，可以选择不画线，或者画出来标上 ns。这里展示全部画出。
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, color='black')
    plt.text((x1+x2)*.5, y+h + 0.01*y_range, text_star, ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')

# 依次画线 (HC_vs_S1, S1_vs_S2, HC_vs_S2)
add_bracket(0, 1, current_y, star_hc_s1)
current_y += step

add_bracket(1, 2, current_y, star_s1_s2)
current_y += step

add_bracket(0, 2, current_y, star_hc_s2)
current_y += step

# 自动调整 y 轴的高度限制，防止顶部的连线被裁剪
plt.ylim(y_min - 0.05 * y_range, current_y + 0.1 * y_range)
# ----------------------------------------

# 动态生成包含主效应 P 值的标题
main_p = anova_table.loc['C(group, Treatment("HC"))', 'PR(>F)']
main_f = anova_table.loc['C(group, Treatment("HC"))', 'F']
title_text = f'Comparison of INT Difference (Max - Min)\nANCOVA (controlling age, sex, mean_fd) Main Effect: $p={main_p:.4e}$'

plt.title(title_text, fontsize=14, pad=15)
plt.ylabel('Difference (Max - Min)', fontsize=12)
plt.xlabel('Group', fontsize=12)

plt.tight_layout()

# 保存并展示图片
save_path = base_dir + 'ThreeGroups_difference_Boxplot_covFD_Stars.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"\n绘图完成！已保存至 {save_path}")