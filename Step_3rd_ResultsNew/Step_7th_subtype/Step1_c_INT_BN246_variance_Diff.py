import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import zscore
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.conftest import matplotlib

# 若不需要在本地弹窗，保留 Agg
matplotlib.use('Agg')

base_dir = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_INTVariance/Step2_BN246_variance/'

# ==========================================
# 1. 读取数据并合并
# ==========================================
df1 = pd.read_csv(base_dir + 'subtype1_BN246_variance.csv')
df2 = pd.read_csv(base_dir + 'subtype2_BN246_variance.csv')

df1['group'] = 'Subtype1'
df2['group'] = 'Subtype2'

# 将两组数据合并，并删除含有缺失值的行
df = pd.concat([df1, df2], ignore_index=True)
df = df.dropna(subset=['variance', 'age', 'sex', 'mean_fd'])

# ==========================================
# 2. 识别并分离原始数据与清洗后数据 (3-Sigma)
# ==========================================
# 按组分别计算 Z-score (自由度 ddof=1)
df['variance_zscore'] = df.groupby('group')['variance'].transform(lambda x: zscore(x, ddof=1))

# 保留一份原始数据的拷贝
df_raw = df.copy()

# 筛选出绝对值 <= 3.0 的数据作为清洗后的数据
df_clean = df[np.abs(df['variance_zscore']) <= 3.0].copy()

print("========== 数据样本量信息 ==========")
print(f"原始数据总样本量: {len(df_raw)}")
print(f"清洗后数据样本量: {len(df_clean)} (共剔除了 {len(df_raw) - len(df_clean)} 个异常值)")
print("====================================\n")


# ==========================================
# 3. 核心统计与绘图函数封装
# ==========================================
def analyze_and_plot(data, suffix, title_note):
    """
    执行 ANCOVA 检验并绘制箱体图
    :param data: 传入的数据框 (df_raw 或 df_clean)
    :param suffix: 文件名后缀 ('Raw' 或 'Cleaned')
    :param title_note: 拼接到图片标题中的说明文字
    """
    print(f"开始处理: 【{suffix}】 数据分析")

    # --- 1. 统计检验：协方差分析 (ANCOVA) ---
    model = ols('variance ~ C(group) + age + C(sex) + mean_fd', data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    p_val = anova_table.loc['C(group)', 'PR(>F)']
    f_val = anova_table.loc['C(group)', 'F']

    print(f"--- ANCOVA 统计结果 ({suffix}) ---")
    print(anova_table)
    print("-" * 40)

    # --- 2. 动态星号标记 ---
    def get_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'ns'

    star_text = get_stars(p_val)

    # --- 3. 绘图对比与显著性标注 ---
    plt.figure(figsize=(8, 6))

    # 绘制箱体图
    sns.boxplot(x='group', y='variance', data=data, order=['Subtype1', 'Subtype2'], palette='Set2', showfliers=True)
    # 叠加散点图
    sns.stripplot(x='group', y='variance', data=data, order=['Subtype1', 'Subtype2'], color='black', alpha=0.4,
                  jitter=True)

    # 获取当前数据的最大最小值，用来按比例计算连线的高度
    y_max = data['variance'].max()
    y_min = data['variance'].min()
    y_range = y_max - y_min

    # 设定线段的高度和向下的"小尾巴"高度
    line_y = y_max + 0.05 * y_range
    h = 0.02 * y_range

    # 画横线及两端的竖线：从 x=0(Subtype1) 到 x=1(Subtype2)
    plt.plot([0, 0, 1, 1], [line_y, line_y + h, line_y + h, line_y], lw=1.5, color='black')
    # 在线段中心偏上方添加星号文字
    plt.text(0.5, line_y + h + 0.01 * y_range, star_text, ha='center', va='bottom', color='black', fontsize=14,
             fontweight='bold')

    # 自动调整 y 轴的高度限制，防止顶部的星号被裁剪掉
    plt.ylim(y_min - 0.05 * y_range, line_y + 0.15 * y_range)

    # 动态生成包含 P 值的标题 (修复了原代码中多余的 (Max - Min) 字眼)
    title_text = f'Comparison of INT Variance\nANCOVA (controlling age, sex, mean_fd)\n{title_note}\n$p={p_val:.4e}$ (F={f_val:.2f})'
    plt.title(title_text, fontsize=14, pad=15)
    plt.ylabel('Variance', fontsize=12)
    plt.xlabel('Subtype', fontsize=12)

    plt.tight_layout()

    # 保存并展示图片
    save_path = base_dir + f'subtype1_subtype2_variance_{suffix}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 核心：必须关闭画布，否则下一张图会叠加在上面

    print(f"[{suffix}] 绘图完成！已保存至 {save_path}\n")


# ==========================================
# 4. 执行分析
# ==========================================
# 第一次运行：原始数据 (Raw)
analyze_and_plot(df_raw, suffix='Raw', title_note='[Original Data]')

# 第二次运行：剔除 >3SD 后的清洗数据 (Cleaned)
analyze_and_plot(df_clean, suffix='Cleaned', title_note='[Outliers > 3SD Removed]')

print("所有处理完毕！")