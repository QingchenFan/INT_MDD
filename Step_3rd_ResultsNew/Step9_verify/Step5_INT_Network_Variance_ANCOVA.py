import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import zscore
from nilearn.conftest import matplotlib
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

# 如果不需要在独立窗口弹出图片，可以使用 Agg 后端
matplotlib.use('Agg')

# 统一定义输出基础路径，方便后续拼写文件名
base_dir = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_verify/Step4_Variance_Diff/'

# =========================================================
# 1. 读取文件与数据合并
# =========================================================
df_hc = pd.read_csv(base_dir + 'HC_network_variances.csv')
df_s1 = pd.read_csv(base_dir + 'subtype1_network_variances.csv')
df_s2 = pd.read_csv(base_dir + 'subtype2_network_variances.csv')

# 提取需要的列 (假设 HC 文件里的 age/sex 列名已经是 age, sex)
df_hc = df_hc[['subID', 'mean_fd', 'age', 'sex', 'variance']]
df_s1 = df_s1[['subID', 'mean_fd', 'age', 'sex', 'variance']]
df_s2 = df_s2[['subID', 'mean_fd', 'age', 'sex', 'variance']]

df_hc['group'] = 'HC'
df_s1['group'] = 'Subtype1'
df_s2['group'] = 'Subtype2'

# 合并得到原始全量数据
df_raw = pd.concat([df_hc, df_s1, df_s2], ignore_index=True)
df_raw = df_raw[['group', 'age', 'sex', 'variance', 'mean_fd']].dropna()

# =========================================================
# 2. 异常值剔除处理 (按 group 分组计算 > 3 SD)
# =========================================================
# 按组计算 Z-score (自由度 ddof=1)
df_raw['variance_zscore'] = df_raw.groupby('group')['variance'].transform(lambda x: zscore(x, ddof=1))

# 设定 3 个标准差为阈值
threshold = 3.0

# 提取绝对值小于等于 3 的数据作为清洗后数据
df_clean = df_raw[np.abs(df_raw['variance_zscore']) <= threshold].copy()

print("========== 数据异常值检查 ==========")
print(f"原始数据样本量: {len(df_raw)}")
print(f"清洗后数据样本量: {len(df_clean)}")
print(f"剔除的异常值数量: {len(df_raw) - len(df_clean)}")
print("====================================\n")


# =========================================================
# 3. 核心统计与绘图函数封装
# =========================================================
def run_analysis_and_plot(df, suffix, title_note):
    """
    执行 ANCOVA, 事后两两比较, 并绘制箱线图
    :param df: 输入的数据框 (df_raw 或 df_clean)
    :param suffix: 输出文件名的后缀 ('Raw' 或 'Clean')
    :param title_note: 图片标题的额外注释
    """
    print(f"开始处理: 【{suffix}】 组数据分析")

    # 1. ANCOVA 主效应模型构建
    model = ols('variance ~ C(group) + age + C(sex) + mean_fd', data=df).fit()

    # 计算协方差分析表 (主效应) 并保存
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_csv_path = base_dir + f'Diff_variance_ANCOVA_MainEffect_covFD_{suffix}.csv'
    anova_table.to_csv(anova_csv_path)

    # 2. 事后检验分析 (Pairwise Comparisons) 与 FDR 校正
    pairwise_results = []
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

    # FDR 校正
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
    pairwise_csv_path = base_dir + f'Diff_variance_ANCOVA_Pairwise_covFD_{suffix}.csv'
    pairwise_df.to_csv(pairwise_csv_path, index=False)

    print(f"[{suffix}] 事后两两比较 (FDR 校正) 结果:")
    print(pairwise_df)
    print("-" * 40)

    # 3. 绘制并动态智能标注显著性的箱体图
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='group', y='variance', data=df, order=['HC', 'Subtype1', 'Subtype2'], palette='Set2')

    y_max = df['variance'].max()
    y_min = df['variance'].min()
    y_range = y_max - y_min

    step = 0.08 * y_range
    h = 0.02 * y_range
    current_y = y_max + 0.05 * y_range

    star_hc_s1 = pairwise_df[pairwise_df['Comparison'] == 'HC_vs_Subtype1']['Significance'].values[0]
    star_s1_s2 = pairwise_df[pairwise_df['Comparison'] == 'Subtype1_vs_Subtype2']['Significance'].values[0]
    star_hc_s2 = pairwise_df[pairwise_df['Comparison'] == 'HC_vs_Subtype2']['Significance'].values[0]

    def add_bracket(x1, x2, y, text_star):
        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color='black')
        plt.text((x1 + x2) * .5, y + h + 0.01 * y_range, text_star, ha='center', va='bottom', color='black',
                 fontsize=12, fontweight='bold')

    if star_hc_s1 != 'ns':
        add_bracket(0, 1, current_y, star_hc_s1)
        current_y += step

    if star_s1_s2 != 'ns':
        add_bracket(1, 2, current_y, star_s1_s2)
        current_y += step

    if star_hc_s2 != 'ns':
        add_bracket(0, 2, current_y, star_hc_s2)
        current_y += step

    plt.ylim(y_min - 0.05 * y_range, current_y + 0.1 * y_range)

    plt.title(f'Comparison of Variance across groups\n(Controlling for Age, Sex & mean_fd)\n{title_note}', fontsize=14,
              pad=15)
    plt.ylabel('Variance', fontsize=12)
    plt.xlabel('Group', fontsize=12)
    plt.tight_layout()

    img_save_path = base_dir + f'Diff_variance_ANCOVA_Stars_covFD_{suffix}.png'
    plt.savefig(img_save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 必须关闭当前图，否则两张图会叠加在同一个画布上

    print(f"=== [{suffix}] 绘图完成！图片已保存至: {img_save_path} ===\n")


# =========================================================
# 4. 运行分析
# =========================================================
# 第一次运行：原始数据 (Raw)
run_analysis_and_plot(df_raw, suffix='Raw', title_note='[Original Data]')

# 第二次运行：剔除 >3SD 后的数据 (Clean)
run_analysis_and_plot(df_clean, suffix='Clean', title_note='[Outliers > 3SD Removed]')

print("所有步骤执行完毕！两套结果（原始的与剔除异常值的）均已分别保存。")