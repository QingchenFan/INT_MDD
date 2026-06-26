import pandas as pd
import numpy as np
import statsmodels.api as sm
from nilearn.conftest import matplotlib
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')
# =========================================================
# 1. 读取文件与数据合并
# =========================================================
base_dir = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_verify/Step2_BN246_INTment_DIff/'
# 注意：请确保以下读取的csv文件中都包含 'mean_fd' 列
df_hc = pd.read_csv(base_dir + 'HC_DZ_INT_agesex_FD.csv')
df_s1 = pd.read_csv(base_dir + 'subtype1_DZ_INT_FD.csv')
df_s2 = pd.read_csv(base_dir + 'subtype2_DZ_INT_FD.csv')

df_hc['group'] = 'HC'
df_s1['group'] = 'Subtype1'
df_s2['group'] = 'Subtype2'

df = pd.concat([df_hc, df_s1, df_s2], ignore_index=True)

# 【修改点 1】：在提取列时，加入 'mean_fd' 并去除缺失值
df = df[['group', 'age', 'sex', 'mean_INT', 'mean_fd']].dropna()

# =========================================================
# 2. ANCOVA 主效应模型构建
# =========================================================
# 【修改点 2】：将 mean_fd 添加到线性回归模型公式中
model = ols('mean_INT ~ C(group) + age + C(sex) + mean_fd', data=df).fit()

# 计算协方差分析表 (主效应) 并保存
anova_table = sm.stats.anova_lm(model, typ=2)
print("========== 协方差分析表 (ANCOVA 主效应) ==========")
print(anova_table)
anova_table.to_csv(base_dir + 'Diff_mean_INT_ANCOVA_MainEffect_covFD.csv') # 更新文件名以示区别

# =========================================================
# 3. 事后检验分析 (Pairwise Comparisons) 与 FDR 校正
# =========================================================
pairwise_results = []
# 这里的 contrast 不变，因为我们只关心 C(group) 的差异，其他协变量 (age, sex, mean_fd) 在对比时保持不变（互相抵消）
contrasts = {
    'HC_vs_Subtype1': 'C(group)[T.Subtype1] = 0',
    'HC_vs_Subtype2': 'C(group)[T.Subtype2] = 0',
    'Subtype1_vs_Subtype2': 'C(group)[T.Subtype1] - C(group)[T.Subtype2] = 0'
}

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

# 对这 3 次两两比较进行 FDR 多重比较校正
_, p_fdr_pair, _, _ = multipletests(pairwise_df['p_uncorrected'], alpha=0.05, method='fdr_bh')
pairwise_df['p_FDR'] = p_fdr_pair

def get_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'

pairwise_df['Significance'] = pairwise_df['p_FDR'].apply(get_stars)

# 保存事后检验详细结果
pairwise_df.to_csv(base_dir + 'Diff_mean_INT_ANCOVA_Pairwise_covFD.csv', index=False)
print("\n========== 事后两两比较 (FDR 校正) ==========")
print(pairwise_df)


# =========================================================
# 4. 绘制并动态智能标注显著性的箱体图
# =========================================================
plt.figure(figsize=(8, 6))
sns.boxplot(x='group', y='mean_INT', data=df, order=['HC', 'Subtype1', 'Subtype2'], palette='Set2')

y_max = df['mean_INT'].max()
y_min = df['mean_INT'].min()
y_range = y_max - y_min

# 连线的动态高度步长和拐角小尾巴长度
step = 0.08 * y_range
h = 0.02 * y_range
current_y = y_max + 0.05 * y_range

# 提取 FDR 校正后的星号
star_hc_s1 = pairwise_df[pairwise_df['Comparison']=='HC_vs_Subtype1']['Significance'].values[0]
star_s1_s2 = pairwise_df[pairwise_df['Comparison']=='Subtype1_vs_Subtype2']['Significance'].values[0]
star_hc_s2 = pairwise_df[pairwise_df['Comparison']=='HC_vs_Subtype2']['Significance'].values[0]

def add_bracket(x1, x2, y, text_star):
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, color='black')
    plt.text((x1+x2)*.5, y+h + 0.01*y_range, text_star, ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')

# 动态堆叠画线逻辑：只有显著的 ('ns' 以外) 才画线
if star_hc_s1 != 'ns':
    add_bracket(0, 1, current_y, star_hc_s1)
    current_y += step

if star_s1_s2 != 'ns':
    add_bracket(1, 2, current_y, star_s1_s2)
    current_y += step

if star_hc_s2 != 'ns':
    add_bracket(0, 2, current_y, star_hc_s2)
    current_y += step

# 自动调整 y 轴范围，为顶部的连线留出空间
plt.ylim(y_min - 0.05 * y_range, current_y + 0.1 * y_range)

# 【修改点 3】：更新标题，体现控制了 FD
plt.title('Comparison of mean_INT across groups\n(Controlling for Age, Sex & mean_fd)', fontsize=14, pad=15)
plt.ylabel('Mean INT', fontsize=12)
plt.xlabel('Group', fontsize=12)
plt.tight_layout()

# 保存并展示图片
img_save_path = base_dir + 'Diff_mean_INT_ANCOVA_Stars_covFD.png'
plt.savefig(img_save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"\n=== 绘图完成！图片已保存至: {img_save_path} ===")