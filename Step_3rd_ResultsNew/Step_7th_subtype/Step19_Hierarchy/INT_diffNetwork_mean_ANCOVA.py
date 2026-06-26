import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os

# ================= 1. 基础配置与函数定义 =================
out_dir = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_NetworkHierarchy/INT_DiffNetwork'
file_hc = os.path.join(out_dir, 'HC_DiffNetwork_mean.csv')
file1 = os.path.join(out_dir, 'subtype1_DiffNetwork_mean.csv')
file2 = os.path.join(out_dir, 'subtype2_DiffNetwork_mean.csv')

# 增加 HC 组的颜色（此处采用了和原配色相近风格的偏绿色调），顺序依次为 HC, Subtype 1, Subtype 2
colors = ['#00A087', '#E64B35', '#4DBBD5']
sns.set_theme(style="ticks", context="talk")


def run_stats_and_plot(data, title_prefix, save_name):
    """
    运行两两配对 OLS 统计分析并绘制包含3组的箱体图
    """
    order = ['HC', 'Subtype 1', 'Subtype 2']

    # 1. 绘图准备
    plt.figure(figsize=(8, 8))

    # 绘制箱体图
    sns.boxplot(
        x='Subtype_Label', y='Mean', data=data, order=order,
        width=0.4, palette=colors, showfliers=False, boxprops=dict(alpha=0.7)
    )

    # 绘制散点图
    sns.stripplot(
        x='Subtype_Label', y='Mean', data=data, order=order,
        color='black', alpha=0.4, jitter=0.15, size=5, zorder=1
    )

    y_max = data['Mean'].max()
    y_min = data['Mean'].min()
    y_range = y_max - y_min

    # 定义需要两两比较的组合
    pairs = [('HC', 'Subtype 1'), ('Subtype 1', 'Subtype 2'), ('HC', 'Subtype 2')]
    print(f"\n[{title_prefix}] OLS 回归统计结果:")

    step = 0.08 * y_range  # 控制每根横线的纵向间距
    line_y_base = y_max + y_range * 0.05

    # 2. 统计分析与添加显著性标记
    for i, (g1, g2) in enumerate(pairs):
        # 提取两组数据
        sub_data = data[data['Subtype_Label'].isin([g1, g2])].copy()

        # 将组别映射为 0 和 1 (dummy variable)
        sub_data['Group_Dummy'] = sub_data['Subtype_Label'].map({g1: 0, g2: 1})

        # 构建模型 (控制年龄、性别、mean_fd)
        Y = sub_data['Mean']
        X = sub_data[['Group_Dummy', 'age', 'sex', 'mean_fd']]
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()

        p_val = model.pvalues['Group_Dummy']
        t_val = model.tvalues['Group_Dummy']

        print(f"{g1} vs {g2} -> T-value: {t_val:.4f}, P-value: {p_val:.4f}")

        # 根据 p 值确定显著性星号
        if p_val < 0.001:
            sig_symbol = '***'
        elif p_val < 0.01:
            sig_symbol = '**'
        elif p_val < 0.05:
            sig_symbol = '*'
        else:
            sig_symbol = 'n.s.'

        # 获取在X轴上的位置
        x1 = order.index(g1)
        x2 = order.index(g2)

        # 画显著性标志线
        line_y = line_y_base + i * step
        plt.plot([x1, x1, x2, x2], [line_y, line_y + y_range * 0.02, line_y + y_range * 0.02, line_y], lw=1.5,
                 color='black')

        # 添加星号/n.s. 文本
        font_size = 14 if sig_symbol == 'n.s.' else 20
        text_y_offset = 0.03 if sig_symbol == 'n.s.' else 0.02
        plt.text((x1 + x2) / 2, line_y + y_range * text_y_offset, sig_symbol, ha='center', va='bottom', color='black',
                 fontsize=font_size)

    # 动态调高Y轴上限，保证三组连线不被截断
    plt.ylim(y_min - y_range * 0.05, line_y_base + 3 * step + y_range * 0.1)

    # 美化
    plt.title(f'{title_prefix}\nMean Difference Network', pad=20, fontsize=15, fontweight='bold')
    plt.ylabel('Mean Difference Value', fontsize=14)
    plt.xlabel('')
    sns.despine()
    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(out_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图片已保存至: {save_path}")


# ================= 2. 加载与预处理原始数据 =================
df_hc = pd.read_csv(file_hc)
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# 添加绘图标签
df_hc['Subtype_Label'] = 'HC'
df1['Subtype_Label'] = 'Subtype 1'
df2['Subtype_Label'] = 'Subtype 2'

# 合并三个组的数据
df_original = pd.concat([df_hc, df1, df2], ignore_index=True)

# ================= 3. 生成原始数据的比较结果 =================
run_stats_and_plot(df_original, "Original Data", 'HC_sub1_sub2_mean_difference_original.png')

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

# 组合三组清理后的数据 (HC保持不变)
df_clean = pd.concat([df_hc, df1_clean, df2], ignore_index=True)

# 生成剔除异常值后的比较结果
run_stats_and_plot(df_clean, "Outliers Removed", 'HC_sub1_sub2_mean_difference_OutliersRemoved.png')