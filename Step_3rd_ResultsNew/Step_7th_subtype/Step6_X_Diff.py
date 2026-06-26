import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 强制使用 Tkinter 后端弹窗，避免部分 IDE 报错 (如果您在 jupyter 运行，请注释掉此行)
matplotlib.use('TkAgg')

# ================= 配置参数区 =================
target_metric = 'QIDS'  # 分析的指标名称

file_path1 = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step11_INT_QIDS_Correlation/subtype1_mean_INT_QIDS.csv"
file_path2 = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step11_INT_QIDS_Correlation/subtype2_mean_INT_QIDS.csv"

group_name1 = 'Subtype 1'
group_name2 = 'Subtype 2'

# 🌟 新增开关：是否剔除异常值 🌟
# 设置为 True: 自动剔除极端异常值后再比较
# 设置为 False: 使用包含所有极端值的原始数据进行比较
REMOVE_OUTLIERS = True
# ==============================================


# 1. 读取数据
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

# 提取目标列并去除空值
data1 = df1[target_metric].dropna()
data2 = df2[target_metric].dropna()

# 记录原始样本量以便后续对比
orig_len1 = len(data1)
orig_len2 = len(data2)

# 2. 异常值剔除逻辑 (IQR 算法)
if REMOVE_OUTLIERS:
    def get_clean_series(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series[(series >= lower_bound) & (series <= upper_bound)]

    data1 = get_clean_series(data1)
    data2 = get_clean_series(data2)
    status_text = "Cleaned Data (Outliers Removed)"
else:
    status_text = "Original Data"


# 3. 统计检验 (稳健的 Mann-Whitney U 检验)
stat, p_val = stats.mannwhitneyu(data1, data2)


# 4. 打印分析报告
print("="*50)
print(f"当前分析状态: {status_text}")
print(f"当前分析指标: {target_metric}")
print("-"*50)

if REMOVE_OUTLIERS:
    print(f"{group_name1} 样本量: {len(data1)} (剔除了 {orig_len1 - len(data1)} 个异常值), 均值: {data1.mean():.2f}")
    print(f"{group_name2} 样本量: {len(data2)} (剔除了 {orig_len2 - len(data2)} 个异常值), 均值: {data2.mean():.2f}")
else:
    print(f"{group_name1} 原始样本量: {len(data1)}, 均值: {data1.mean():.2f}")
    print(f"{group_name2} 原始样本量: {len(data2)}, 均值: {data2.mean():.2f}")

print(f"Mann-Whitney U 检验 p 值: {p_val:.4f}")
if p_val < 0.05:
    print("结论: 两组之间存在显著差异 (*p < 0.05)")
else:
    print("结论: 两组之间没有显著差异 (p >= 0.05)")
print("="*50)


# ==================== 绘制箱体图 ====================
# 整合用于画图的数据
plot_data = pd.DataFrame({
    target_metric: list(data1) + list(data2),
    'Group': [group_name1] * len(data1) + [group_name2] * len(data2)
})

# 设置单张图形大小
plt.figure(figsize=(8, 6))

# 绘制箱线图
sns.boxplot(x='Group', y=target_metric, data=plot_data, width=0.4, palette='Set2')
# 叠加散点
sns.stripplot(x='Group', y=target_metric, data=plot_data, color='black', alpha=0.3, jitter=True)

# 动态生成标题和坐标轴
plt.title(f'{target_metric} Scores: {status_text}\n(p = {p_val:.4f})', fontsize=14)
plt.ylabel(f'{target_metric} Scores', fontsize=12)
plt.xlabel('', fontsize=12)

# 增加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 自动调整布局并展示
plt.tight_layout()

# 保存文件命名也会跟随开关动态改变
save_filename = f"{target_metric}_{'Cleaned' if REMOVE_OUTLIERS else 'Original'}_Comparison.png"
plt.savefig(save_filename, dpi=300)
print(f"✅ 图表已保存至: {save_filename}")

plt.show()