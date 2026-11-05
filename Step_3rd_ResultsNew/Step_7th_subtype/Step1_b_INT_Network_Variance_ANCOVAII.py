import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib


matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# ==========================================
# 1. 数据读取与每组独立计算 Variance
# ==========================================
base_dir = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_FirstRe/Episode/'
df_hc = pd.read_csv(f'{base_dir}/HC_INT20_7net_agesex_FD.csv')
df_sub1 = pd.read_csv(f'{base_dir}/subtype1_INT_7net_agesex.csv')
df_sub2 = pd.read_csv(f'{base_dir}/subtype2_INT_7net_agesex.csv')

# 定义用来计算方差的 8 个网络
networks = ['subcortical', 'Visual', 'Somatomotor', 'Dorsal_Attention',
            'Ventral_Attention', 'Limbic', 'Frontoparietal', 'Default']

# 核心步骤：分别计算每组中每个人的网络间 Variance
df_hc['network_variance'] = df_hc[networks].var(axis=1)
df_sub1['network_variance'] = df_sub1[networks].var(axis=1)
df_sub2['network_variance'] = df_sub2[networks].var(axis=1)

# ==========================================
# 2. 将计算得到的 variance 与指定原变量一起保存为新 CSV
# ==========================================
save_cols = ['subID', 'age', 'sex', 'mean_fd', 'network_variance']

# 分别保存回原始路径（文件名添加了 _with_variance 后缀）
df_hc[save_cols].to_csv(f'{base_dir}/HC_INT20_7net_agesex_FD_with_variance.csv', index=False)
df_sub1[save_cols].to_csv(f'{base_dir}/subtype1_INT_7net_agesex_FD_with_variance.csv', index=False)
df_sub2[save_cols].to_csv(f'{base_dir}/subtype2_INT_7net_agesex_FD_with_variance.csv', index=False)

print(" [系统提示] 各组包含 network_variance 的新 CSV 文件已成功保存至原目录。")

# ==========================================
# 3. 合并全集数据并转换协变量类型
# ==========================================
# 添加分组标签以便后续合并比较
df_hc['Group'] = 'HC'
df_sub1['Group'] = 'Subtype1'
df_sub2['Group'] = 'Subtype2'

# 合并所有数据
df_all = pd.concat([df_hc, df_sub1, df_sub2], ignore_index=True)

# 确保协变量的数据类型正确
df_all['sex'] = df_all['sex'].astype(str)
df_all['age'] = df_all['age'].astype(float)
df_all['mean_fd'] = df_all['mean_fd'].astype(float)


# ==========================================
# 4. 封装绘图及统计标记函数
# ==========================================
def add_stat_annotation(ax, data, x, y, order):
    """进行两两独立样本 t 检验并添加显著性标记 *"""
    groups = order
    pairs = list(combinations(groups, 2))

    y_max = data[y].max()
    y_range = data[y].max() - data[y].min()

    h = y_range * 0.03
    text_h = y_range * 0.01
    current_y = y_max + h

    for pair in pairs:
        group1_data = data[data[x] == pair[0]][y]
        group2_data = data[data[x] == pair[1]][y]

        t_stat, p_val = stats.ttest_ind(group1_data, group2_data, nan_policy='omit')

        if p_val < 0.001:
            sig_symbol = '***'
        elif p_val < 0.01:
            sig_symbol = '**'
        elif p_val < 0.05:
            sig_symbol = '*'
        else:
            sig_symbol = 'ns'

        if sig_symbol != 'ns':
            x1 = groups.index(pair[0])
            x2 = groups.index(pair[1])

            ax.plot([x1, x1, x2, x2], [current_y, current_y + h, current_y + h, current_y], lw=1.2, c='k')
            ax.text((x1 + x2) * .5, current_y + h + text_h, sig_symbol, ha='center', va='bottom', color='k',
                    fontsize=12)
            current_y += y_range * 0.1


def make_boxplot(data, title, save_filename):
    """画箱体图并保存"""
    group_order = ['HC', 'Subtype1', 'Subtype2']
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(x='Group', y='network_variance', data=data, order=group_order,
                     palette="Set2", width=0.5, fliersize=0)
    sns.stripplot(x='Group', y='network_variance', data=data, order=group_order,
                  color=".25", alpha=0.5, jitter=True, size=4)

    add_stat_annotation(ax, data, x='Group', y='network_variance', order=group_order)

    plt.title(title, fontsize=13, fontweight='bold')
    plt.ylabel('Variance across 8 Networks', fontsize=12)
    plt.xlabel('Group', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300)
    plt.close()


# 定义 ANCOVA 统一公式
formula = "network_variance ~ C(Group) + age + C(sex) + mean_fd"

# ==========================================
# 5. 分析一：原始 Variance 差异比较 (未剔除异常值)
# ==========================================
print(f"\n{'=' * 50}")
print(" 分析一：原始 Variance 差异比较（包含所有个体）")
print(f"{'=' * 50}")
print(f"总初始样本量: {len(df_all)}")

# 运行原始数据的 ANCOVA 模型
model_orig = ols(formula, data=df_all).fit()
anova_table_orig = sm.stats.anova_lm(model_orig, typ=2)
print("\n[结果] 原始数据 ANCOVA 表:")
print(anova_table_orig)

# 绘制并保存原始数据的箱体图
make_boxplot(df_all, 'Original Network Variance across Groups', 'Network_Variance_Original.png')
print(" [系统提示] 原始数据箱体图已成功保存为 'Network_Variance_Original.png'")

# ==========================================
# 6. 分析二：剔除异常值 ( > 3 SD ) 后的差异比较
# ==========================================
print(f"\n{'=' * 50}")
print(" 分析二：剔除大于 3 个标准差（3SD）后的 Variance 差异比较")
print(f"{'=' * 50}")

# 基于整体的 network_variance 计算 Z-score
z_scores = np.abs(stats.zscore(df_all['network_variance'], nan_policy='omit'))

# 筛选 3 个标准差以内的数据
df_clean = df_all[z_scores <= 3].copy()
removed_count = len(df_all) - len(df_clean)

print(f"剔除了 {removed_count} 个异常值 ( Variance > 3 SD )")
print(f"最终清洗后纳入分析样本量: {len(df_clean)}")

# 运行清洗后数据的 ANCOVA 模型
model_clean = ols(formula, data=df_clean).fit()
anova_table_clean = sm.stats.anova_lm(model_clean, typ=2)
print("\n[结果] 剔除异常值后的 ANCOVA 表:")
print(anova_table_clean)

# 绘制并保存清洗后数据的箱体图
make_boxplot(df_clean, 'Network Variance across Groups (Outliers > 3SD removed)', 'Network_Variance_Cleaned.png')
print(" [系统提示] 清洗后数据箱体图已成功保存为 'Network_Variance_Cleaned.png'")