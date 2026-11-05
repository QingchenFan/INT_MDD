import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os

# ================= 1. 基础配置与函数定义 =================
out_dir = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_NetworkHierarchy/INT_DiffNetwork'
file1 = os.path.join(out_dir, 'subtype1_DiffNetwork_mean.csv')
file2 = os.path.join(out_dir, 'subtype2_DiffNetwork_mean.csv')
colors = ['#E64B35', '#4DBBD5']
sns.set_theme(style="ticks", context="talk")


def run_stats_and_plot(data, title_prefix, save_name):
    """
    运行 OLS 统计分析并绘制箱体图
    """
    # 1. 统计分析 (控制年龄、性别、mean_fd)
    Y = data['Mean']  # 注意这里是大写 Mean
    X = data[['Group', 'age', 'sex', 'mean_fd']]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    p_val = model.pvalues['Group']
    t_val = model.tvalues['Group']

    print(f"\n[{title_prefix}] OLS 回归统计结果 (Sub1 vs Sub2):")
    print(f"T-value: {t_val:.4f}, P-value: {p_val:.4f}")

    # 根据 p 值确定显著性星号
    if p_val < 0.001:
        sig_symbol = '***'
    elif p_val < 0.01:
        sig_symbol = '**'
    elif p_val < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = 'n.s.'  # n.s. 表示不显著

    # 2. 绘图
    plt.figure(figsize=(6, 8))

    # 绘制箱体图 (统一使用大写的 'Mean')
    sns.boxplot(
        x='Subtype_Label', y='Mean', data=data,
        width=0.4, palette=colors, showfliers=False, boxprops=dict(alpha=0.7)
    )

    # 绘制散点图
    sns.stripplot(
        x='Subtype_Label', y='Mean', data=data,
        color='black', alpha=0.4, jitter=0.15, size=5, zorder=1
    )

    # 添加统计学显著性连线和标记
    y_max = data['Mean'].max()
    y_min = data['Mean'].min()
    y_range = y_max - y_min
    line_y = y_max + y_range * 0.05

    plt.plot([0, 0, 1, 1], [line_y, line_y + y_range * 0.02, line_y + y_range * 0.02, line_y], lw=1.5, color='black')

    # 根据是否显著调整文本高度和大小
    font_size = 14 if sig_symbol == 'n.s.' else 20
    text_y_offset = 0.03 if sig_symbol == 'n.s.' else 0.02
    plt.text(0.5, line_y + y_range * text_y_offset, sig_symbol, ha='center', va='bottom', color='black',
             fontsize=font_size)

    # 美化
    plt.title(f'{title_prefix}\nMean Difference Network', pad=20, fontsize=15, fontweight='bold')
    plt.ylabel('Mean Difference Value', fontsize=14)
    plt.xlabel('')
    sns.despine()
    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(out_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭当前图表，避免重叠
    print(f"图片已保存至: {save_path}")


# ================= 2. 加载与预处理原始数据 =================
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# 添加绘图标签和统计用的 Group 列 (防御性编程，确保有Group列)
df1['Subtype_Label'] = 'Subtype 1'
df2['Subtype_Label'] = 'Subtype 2'
if 'Group' not in df1.columns: df1['Group'] = 1
if 'Group' not in df2.columns: df2['Group'] = 0

df_original = pd.concat([df1, df2], ignore_index=True)

# ================= 3. 生成原始数据的比较结果 =================
run_stats_and_plot(df_original, "Original Data", 'mean_difference_boxplot_original.png')

# ================= 4. 剔除 Subtype1 异常值并生成新结果 =================
# 使用 IQR 方法剔除 df1 (Subtype 1) 的异常值 (统一使用大写的 'Mean')
Q1 = df1['Mean'].quantile(0.25)
Q3 = df1['Mean'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 筛选出正常范围内的数据
df1_clean = df1[(df1['Mean'] >= lower_bound) & (df1['Mean'] <= upper_bound)].copy()

print("-" * 40)
print(f"异常值处理报告：")
print(f"Subtype 1 原有被试数: {len(df1)}")
print(f"剔除异常值后被试数:   {len(df1_clean)}")
print(f"共移除了 {len(df1) - len(df1_clean)} 个离群点。")
print("-" * 40)

# 组合清理后的数据
df_clean = pd.concat([df1_clean, df2], ignore_index=True)

# 生成剔除异常值后的比较结果
run_stats_and_plot(df_clean, "Outliers Removed", 'mean_difference_boxplot_cleaned.png')