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
path_hc = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step3_bGMV7Net_Diff/HC_GMV_7net_agesex.csv'
path_s1 = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step3_bGMV7Net_Diff/subtype1_GMV_7Net.csv'
path_s2 = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step3_bGMV7Net_Diff/subtype2_GMV_7Net.csv'

df_hc = pd.read_csv(path_hc)
df_s1 = pd.read_csv(path_s1)
df_s2 = pd.read_csv(path_s2)

# 添加组别标签
df_hc['group'] = 'HC'
df_s1['group'] = 'Subtype1'
df_s2['group'] = 'Subtype2'

# 合并大表并提取所需的列（加入 TIV 作为协变量）
df = pd.concat([df_hc, df_s1, df_s2], ignore_index=True)
networks = ['subcortical_GMV', 'Visual_GMV', 'Somatomotor_GMV', 'Dorsal_Attention_GMV',
            'Ventral_Attention_GMV', 'Limbic_GMV', 'Frontoparietal_GMV', 'Default_GMV']
df = df[['group', 'age', 'sex', 'TIV'] + networks].dropna()

# =========================================================
# 2. ANCOVA 主效应分析与事后两两比较计算
# =========================================================
results = []
pairwise_results = []
models = {}

for net in networks:
    # 构建 ANCOVA 模型: 控制 TIV, age, sex
    formula = f'{net} ~ C(group) + age + C(sex) + TIV'
    model = ols(formula, data=df).fit()
    models[net] = model

    # 提取 ANOVA 表格 (主效应)
    anova_table = sm.stats.anova_lm(model, typ=2)
    p_group = anova_table.loc['C(group)', 'PR(>F)']
    f_group = anova_table.loc['C(group)', 'F']
    net_clean_name = net.replace('_GMV', '')

    results.append({
        'Network': net_clean_name,
        'F_value': f_group,
        'p_uncorrected': p_group
    })

    # 构建事后两两比较的 Contrast
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
            'Network': net_clean_name,
            'Comparison': comp_name,
            't_value': t_val,
            'p_uncorrected': p_val
        })

# --- 主效应 FDR 校正 ---
results_df = pd.DataFrame(results)
_, p_fdr_main, _, _ = multipletests(results_df['p_uncorrected'], alpha=0.05, method='fdr_bh')
results_df['p_FDR'] = p_fdr_main
# 保存主效应结果
results_df.to_csv(
    '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step3_bGMV7Net_Diff/ANCOVA_8networks_GMV_MainEffect.csv',
    index=False)

# --- 事后两两比较 FDR 校正 (对24次检验进行全局校正) ---
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
# 保存两两比较结果
pairwise_df.to_csv(
    '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step3_bGMV7Net_Diff/ANCOVA_8networks_GMV_Pairwise.csv',
    index=False)

print("=== 核心计算完成：主效应与两两比较已分别进行 FDR 校正并保存 ===")

# =========================================================
# 3. 绘制并动态智能标注显著性的箱体图
# =========================================================
df_melted = df.melt(id_vars=['group', 'age', 'sex', 'TIV'], value_vars=networks,
                    var_name='Network', value_name='GMV')
df_melted['Network'] = df_melted['Network'].str.replace('_GMV', '')

plt.figure(figsize=(16, 7))
ax = sns.boxplot(x='Network', y='GMV', hue='group', data=df_melted,
                 hue_order=['HC', 'Subtype1', 'Subtype2'], palette='Set3')

# 基础绘图参数设定
y_range = df_melted['GMV'].max() - df_melted['GMV'].min()
step = 0.06 * y_range  # 连线每多一层，向上偏移的距离
h = 0.015 * y_range  # 倒 U 型线垂直向下的小尾巴长度

# 循环 8 个网络，分别标星号
for i, net_name in enumerate(results_df['Network']):
    # 提取该脑区内三组两两比较的校正 p 值星号
    star_hc_s1 = pairwise_df[(pairwise_df['Network'] == net_name) & (pairwise_df['Comparison'] == 'HC_vs_Subtype1')][
        'Significance'].values[0]
    star_s1_s2 = \
    pairwise_df[(pairwise_df['Network'] == net_name) & (pairwise_df['Comparison'] == 'Subtype1_vs_Subtype2')][
        'Significance'].values[0]
    star_hc_s2 = pairwise_df[(pairwise_df['Network'] == net_name) & (pairwise_df['Comparison'] == 'HC_vs_Subtype2')][
        'Significance'].values[0]

    # 计算当前网络内箱体的相对 x 坐标 (seaborn 默认偏移量)
    x_hc = i - 0.26
    x_s1 = i
    x_s2 = i + 0.26

    # 找到该网络的最高值，作为标注画线的基准起点，防止线画得太远或嵌入图形中
    net_max = df_melted[df_melted['Network'] == net_name]['GMV'].max()
    current_y = net_max + 0.03 * y_range  # 初始化起始高度


    # 定义画线内部函数
    def add_bracket(x1, x2, y, text_star):
        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color='black')
        plt.text((x1 + x2) * .5, y + h, text_star, ha='center', va='bottom', color='black', fontsize=12,
                 fontweight='bold')


    # 动态堆叠画线逻辑：只有显著的 ('ns' 以外) 才画线，且画完后高度自动抬升，防止交叉
    if star_hc_s1 != 'ns':
        add_bracket(x_hc, x_s1, current_y, star_hc_s1)
        current_y += step

    if star_s1_s2 != 'ns':
        add_bracket(x_s1, x_s2, current_y, star_s1_s2)
        current_y += step

    if star_hc_s2 != 'ns':
        add_bracket(x_hc, x_s2, current_y, star_hc_s2)
        current_y += step

# 留出图表顶端空间，防止星号被图表边缘裁掉
plt.ylim(df_melted['GMV'].min() - 0.05 * y_range, df_melted['GMV'].max() + 0.28 * y_range)

plt.title('Comparison of 8 Networks GMV across groups (Covariates: Age, Sex, TIV)', fontsize=14, pad=15)
plt.ylabel('Gray Matter Volume (GMV)', fontsize=12)
plt.xlabel('Network', fontsize=12)
plt.xticks(rotation=15, fontsize=11)

# 将图例移到侧边，避免遮挡数据
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()

# 保存高清图片
save_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step3_bGMV7Net_Diff/ANCOVA_8networks_GMV_Boxplot_Stars.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"=== 绘图完成！图片已保存至: {save_path} ===")