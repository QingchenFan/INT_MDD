import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import zscore
from nilearn.conftest import matplotlib
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

# 若不需要在本地直接弹窗，可以使用 Agg 后端
matplotlib.use('Agg')

# ==========================================
# 1. 统一定义输出基础路径
# ==========================================
base_dir = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_test/Step2_subtype/Step4_b_INTRange/'

# 读取数据
df_hc = pd.read_csv(base_dir + 'HC_net_range.csv')
df1 = pd.read_csv(base_dir + 'subtype1_INT8Net_Range.csv')
df2 = pd.read_csv(base_dir + 'subtype2_INT8Net_Range.csv')

# ==========================================
# 2. 标签处理与数据合并
# ==========================================
df_hc['group'] = 'HC'
df1['group'] = 'Subtype1'
df2['group'] = 'Subtype2'

# 合并得到原始全量数据
df_raw = pd.concat([df_hc, df1, df2], ignore_index=True)
df_raw = df_raw.dropna(subset=['difference', 'age', 'sex', 'mean_fd'])

# 将 group 转换为有顺序的分类变量
df_raw['group'] = pd.Categorical(df_raw['group'], categories=['HC', 'Subtype1', 'Subtype2'])

# ==========================================
# 3. 异常值剔除处理 (按 group 分组计算 > 3 SD)
# ==========================================
# 按组计算 Z-score (自由度 ddof=1)
df_raw['difference_zscore'] = df_raw.groupby('group')['difference'].transform(lambda x: zscore(x, ddof=1))

# 设定 3 个标准差为阈值
threshold = 3.0
df_clean = df_raw[np.abs(df_raw['difference_zscore']) <= threshold].copy()

print("========== 数据异常值检查 ==========")
print(f"原始数据样本量: {len(df_raw)}")
print(f"清洗后数据样本量: {len(df_clean)}")
print(f"剔除的异常值数量: {len(df_raw) - len(df_clean)}")
print("====================================\n")


# ==========================================
# 4. 核心统计与绘图函数封装
# ==========================================
def run_analysis_and_plot(df, suffix, title_note):
    print(f"开始处理: 【{suffix}】 组数据分析 (INT Range/Difference)")

    # 1. ANCOVA 主效应模型构建
    # 使用 Treatment("HC") 明确将 HC 作为基线对照组
    model = ols('difference ~ C(group, Treatment("HC")) + age + C(sex) + mean_fd', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # 保存主效应结果到 CSV
    anova_csv_path = base_dir + f'Diff_range_ANCOVA_MainEffect_covFD_{suffix}.csv'
    anova_table.to_csv(anova_csv_path)

    # 2. 事后检验分析 (Pairwise Comparisons)
    pairwise_results = []
    contrasts = {
        'HC_vs_Subtype1': 'C(group, Treatment("HC"))[T.Subtype1] = 0',
        'HC_vs_Subtype2': 'C(group, Treatment("HC"))[T.Subtype2] = 0',
        'Subtype1_vs_Subtype2': 'C(group, Treatment("HC"))[T.Subtype1] - C(group, Treatment("HC"))[T.Subtype2] = 0'
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
    _, p_fdr, _, _ = multipletests(pairwise_df['p_uncorrected'], alpha=0.05, method='fdr_bh')
    pairwise_df['p_FDR'] = p_fdr

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

    # 保存事后检验结果到 CSV
    pairwise_csv_path = base_dir + f'Diff_range_ANCOVA_Pairwise_covFD_{suffix}.csv'
    pairwise_df.to_csv(pairwise_csv_path, index=False)

    print(f"[{suffix}] 事后两两比较 (FDR 校正) 结果:")
    print(pairwise_df)
    print("-" * 40)

    # 3. 绘图与显著性标注
    plt.figure(figsize=(9, 6))

    # 绘制箱体图 + 散点图
    sns.boxplot(x='group', y='difference', data=df, order=['HC', 'Subtype1', 'Subtype2'], palette='Set2',
                showfliers=True)
    sns.stripplot(x='group', y='difference', data=df, order=['HC', 'Subtype1', 'Subtype2'], color='black', alpha=0.3,
                  jitter=True)

    y_max = df['difference'].max()
    y_min = df['difference'].min()
    y_range = y_max - y_min

    step = 0.08 * y_range
    h = 0.02 * y_range
    current_y = y_max + 0.05 * y_range

    star_hc_s1 = pairwise_df[pairwise_df['Comparison'] == 'HC_vs_Subtype1']['Significance'].values[0]
    star_hc_s2 = pairwise_df[pairwise_df['Comparison'] == 'HC_vs_Subtype2']['Significance'].values[0]
    star_s1_s2 = pairwise_df[pairwise_df['Comparison'] == 'Subtype1_vs_Subtype2']['Significance'].values[0]

    def add_bracket(x1, x2, y, text_star):
        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color='black')
        plt.text((x1 + x2) * .5, y + h + 0.01 * y_range, text_star, ha='center', va='bottom', color='black',
                 fontsize=12, fontweight='bold')

    # 动态画线逻辑：如果不显著 (ns) 就不画线；如果您想全都画出来，请将 if 判断去掉
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

    # 动态生成标题
    main_p = anova_table.loc['C(group, Treatment("HC"))', 'PR(>F)']
    title_text = f'Comparison of INT Difference (Max - Min)\nANCOVA (cov FD) Main Effect: $p={main_p:.4e}$\n{title_note}'

    plt.title(title_text, fontsize=14, pad=15)
    plt.ylabel('Difference (Max - Min)', fontsize=12)
    plt.xlabel('Group', fontsize=12)
    plt.tight_layout()

    # 保存图片并关闭画板
    img_save_path = base_dir + f'Diff_range_ANCOVA_Stars_covFD_{suffix}.png'
    plt.savefig(img_save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"=== [{suffix}] 绘图完成！图片已保存至: {img_save_path} ===\n")


# ==========================================
# 5. 运行分析
# ==========================================
# 第一次运行：原始数据 (Raw)
run_analysis_and_plot(df_raw, suffix='Raw', title_note='[Original Data]')

# 第二次运行：剔除 >3SD 后的数据 (Clean)
run_analysis_and_plot(df_clean, suffix='Clean', title_note='[Outliers > 3SD Removed]')

print("所有步骤执行完毕！主效应表、事后检验结果表以及可视化图表均已成功导出。")