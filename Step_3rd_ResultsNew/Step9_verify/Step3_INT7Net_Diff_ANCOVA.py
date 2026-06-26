import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 1. 数据读取与预处理
# =========================================================
# ⚠️ 注意：请确保这三个原始 csv 文件中都包含了 'mean_fd' 这一列。
# 如果不包含，请先将 mean_fd 合并到这三个输入文件中。
path_hc = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_verify/HC_DZ_INT_8net.csv'
path_s1 = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_verify/subtype1_DZ_INT_8net.csv'
path_s2 = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_verify/subtype2_DZ_INT_8net.csv'

df_hc = pd.read_csv(path_hc)
df_s1 = pd.read_csv(path_s1)
df_s2 = pd.read_csv(path_s2)

# 添加组别标签
df_hc['group'] = 'HC'
df_s1['group'] = 'Subtype1'
df_s2['group'] = 'Subtype2'

# 合并大表并提取所需的列
df = pd.concat([df_hc, df_s1, df_s2], ignore_index=True)
networks = ['Subcortical', 'Visual', 'Somatomotor', 'Dorsal_Attention',
            'Ventral_Attention', 'Limbic', 'Frontoparietal', 'Default']

# 【修改点1】：提取列时加入 mean_fd 字段
df = df[['group', 'age', 'sex', 'mean_fd'] + networks].dropna()

# =========================================================
# 2. ANCOVA 主效应分析与事后两两比较计算
# =========================================================
results = []
pairwise_results = []
models = {}

for net in networks:
    # 【修改点2】：模型构建：控制 age, sex, mean_fd
    formula = f'{net} ~ C(group) + age + C(sex) + mean_fd'
    model = ols(formula, data=df).fit()
    models[net] = model

    # 提取 ANOVA 表格 (主效应)
    anova_table = sm.stats.anova_lm(model, typ=2)
    p_group = anova_table.loc['C(group)', 'PR(>F)']
    f_group = anova_table.loc['C(group)', 'F']

    results.append({
        'Network': net,
        'F_value': f_group,
        'p_uncorrected': p_group
    })

    # 构建事后两两比较的 Contrast (对每一对组别在控制了协变量后的估算均值进行检验)
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
            'Network': net,
            'Comparison': comp_name,
            't_value': t_val,
            'p_uncorrected': p_val
        })

# --- 主效应 FDR 校正 (8次检验) ---
results_df = pd.DataFrame(results)
_, p_fdr_main, _, _ = multipletests(results_df['p_uncorrected'], alpha=0.05, method='fdr_bh')
results_df['p_FDR'] = p_fdr_main

# 【修改点3】：保存主效应结果（更改文件名避免覆盖）
results_df.to_csv(
    '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_verify/Step2_8Net_INT_mean_Diff/ANCOVA_8networks_INT_MainEffect_covFD.csv',
    index=False)

# --- 事后两两比较 FDR 校正 (全局24次检验) ---
pairwise_df = pd.DataFrame(pairwise_results)
_, p_fdr_pair, _, _ = multipletests(pairwise_df['p_uncorrected'], alpha=0.05, method='fdr_bh')
pairwise_df['p_FDR'] = p_fdr_pair


def get_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


pairwise_df['Significance'] = pairwise_df['p_FDR'].apply(get_stars)

# 【修改点3】：保存两两比较结果（更改文件名避免覆盖）
pairwise_df.to_csv(
    '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_verify/Step2_8Net_INT_mean_Diff/ANCOVA_8networks_INT_Pairwise_covFD.csv',
    index=False)

print("=== INT 主效应与两两比较分析完成！已控制 Age, Sex, mean_fd，进行 FDR 校正并保存至 CSV ===")

# =========================================================
# 3. 绘制并动态智能标注显著性的箱体图
# =========================================================
# 【修改点4】：宽表变长表交给 seaborn 时，保留 mean_fd 作为标识变量之一（虽然绘图用不到，但保持结构一致）
df_melted = df.melt(id_vars=['group', 'age', 'sex', 'mean_fd'], value_vars=networks,
                    var_name='Network', value_name='INT_value')

plt.figure(figsize=(16, 7))
ax = sns.boxplot(x='Network', y='INT_value', hue='group', data=df_melted,
                 hue_order=['HC', 'Subtype1', 'Subtype2'], palette='Set2')

# 基础绘图参数设定，用于动态绘制星号连线
y_range = df_melted['INT_value'].max() - df_melted['INT_value'].min()
step = 0.06 * y_range  # 连线堆叠高度
h = 0.015 * y_range  # 连线下拐的小尾巴长度

# 循环 8 个网络，分别标星号
for i, net_name in enumerate(results_df['Network']):
    # 提取当前网络的三组两两比较星号结果
    star_hc_s1 = pairwise_df[(pairwise_df['Network'] == net_name) & (pairwise_df['Comparison'] == 'HC_vs_Subtype1')]['Significance'].values[0]
    star_s1_s2 = pairwise_df[(pairwise_df['Network'] == net_name) & (pairwise_df['Comparison'] == 'Subtype1_vs_Subtype2')]['Significance'].values[0]
    star_hc_s2 = pairwise_df[(pairwise_df['Network'] == net_name) & (pairwise_df['Comparison'] == 'HC_vs_Subtype2')]['Significance'].values[0]

    # 确定各组箱体在 X 轴的相对坐标 (Seaborn default offset for 3 hue levels)
    x_hc = i - 0.26
    x_s1 = i
    x_s2 = i + 0.26

    # 找到该网络的最高值，作为标注画线的基准起点
    net_max = df_melted[df_melted['Network'] == net_name]['INT_value'].max()
    current_y = net_max + 0.03 * y_range

    def add_bracket(x1, x2, y, text_star):
        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color='black')
        plt.text((x1 + x2) * .5, y + h, text_star, ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')

    # 仅绘制存在显著差异的连线，并自动向上堆叠避开重叠
    if star_hc_s1 != 'ns':
        add_bracket(x_hc, x_s1, current_y, star_hc_s1)
        current_y += step

    if star_s1_s2 != 'ns':
        add_bracket(x_s1, x_s2, current_y, star_s1_s2)
        current_y += step

    if star_hc_s2 != 'ns':
        add_bracket(x_hc, x_s2, current_y, star_hc_s2)
        current_y += step

# 自动留出上方空白区域，避免星号超出图框
plt.ylim(df_melted['INT_value'].min() - 0.05 * y_range, df_melted['INT_value'].max() + 0.28 * y_range)

# 【修改点5】：更新图表标题，展示已经控制了 mean_fd
plt.title('Comparison of 8 Networks INT across groups\n(Covariates: Age, Sex, mean_fd)', fontsize=14, pad=15)
plt.ylabel('INT Value', fontsize=12)
plt.xlabel('Network', fontsize=12)
plt.xticks(rotation=15, fontsize=11)

# 优化图例位置防遮盖
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()

# 【修改点3】：保存带星号的高清图（更改文件名避免覆盖）
save_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_verify/Step2_8Net_INT_mean_Diff/ANCOVA_8networks_INT_Boxplot_Stars_covFD.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
#plt.show()

print(f"=== 绘图完成！图片已保存至: {save_path} ===")