import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 强制使用 Tkinter 后端弹窗，避免部分 IDE 报错 (如果您在 jupyter 运行，请注释掉此行)
matplotlib.use('TkAgg')

# ================= 配置参数区 =================
target_metric = 'QIDS'  # 分析的指标名称

file_path1 = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_Match/Step2_subtype/Step6_QIDS_Diff/subtype1_QIDS.csv"
file_path2 = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_Match/Step2_subtype/Step6_QIDS_Diff/subtype2_QIDS.csv"

group_name1 = 'Subtype 1'
group_name2 = 'Subtype 2'

# 新增开关：是否剔除异常值 (IQR法)
REMOVE_OUTLIERS = False
# ==============================================

# 1. 读取数据并打上组别标签
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

df1['Group'] = group_name1
df2['Group'] = group_name2

# 提取所需列并去除含有空值的行，确保统计模型能正常运行
cols_needed = [target_metric, 'Group', 'age', 'sex']
df1 = df1.dropna(subset=[target_metric, 'age', 'sex'])
df2 = df2.dropna(subset=[target_metric, 'age', 'sex'])

orig_len1 = len(df1)
orig_len2 = len(df2)

# 2. 异常值剔除逻辑 (IQR 算法) - 按组别独立剔除
if REMOVE_OUTLIERS:
    def get_clean_df(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    df1_clean = get_clean_df(df1, target_metric)
    df2_clean = get_clean_df(df2, target_metric)
    status_text = "Cleaned Data (Outliers Removed)"
else:
    df1_clean = df1.copy()
    df2_clean = df2.copy()
    status_text = "Original Data"

# 合并数据供 ANCOVA 和画图使用
df_all = pd.concat([df1_clean, df2_clean], ignore_index=True)

# 确保数据类型正确
df_all['age'] = pd.to_numeric(df_all['age'], errors='coerce')
df_all['sex'] = df_all['sex'].astype(str)
df_all['Group'] = df_all['Group'].astype('category')

# 3. 统计检验: 协方差分析 (ANCOVA) 控制 Age 和 Sex
# 公式: 目标指标 ~ 组别 + 年龄 + 性别
formula = f'{target_metric} ~ C(Group) + age + C(sex)'
model = smf.ols(formula, data=df_all).fit()
anova_results = anova_lm(model, typ=2)

# 提取 Group 的 p 值
p_val = anova_results.loc['C(Group)', 'PR(>F)']
f_stat = anova_results.loc['C(Group)', 'F']

ss_group = anova_results.loc['C(Group)', 'sum_sq']
ss_resid = anova_results.loc['Residual', 'sum_sq']
eta_p2 = ss_group / (ss_group + ss_resid)
print(f"偏η² (partial eta squared): {eta_p2:.4f}")




# 4. 打印分析报告
print("="*50)
print(f"当前分析状态: {status_text}")
print(f"当前分析指标: {target_metric}")
print("-" * 50)

if REMOVE_OUTLIERS:
    print(f"{group_name1} 样本量: {len(df1_clean)} (剔除 {orig_len1 - len(df1_clean)} 个), 均值: {df1_clean[target_metric].mean():.2f}")
    print(f"{group_name2} 样本量: {len(df2_clean)} (剔除 {orig_len2 - len(df2_clean)} 个), 均值: {df2_clean[target_metric].mean():.2f}")
else:
    print(f"{group_name1} 原始样本量: {len(df1_clean)}, 均值: {df1_clean[target_metric].mean():.2f}")
    print(f"{group_name2} 原始样本量: {len(df2_clean)}, 均值: {df2_clean[target_metric].mean():.2f}")

print("-" * 50)
print("【 ANCOVA 统计结果 (已控制年龄和性别) 】")
print(f"F 统计量: {f_stat:.4f}")
print(f"p 值: {p_val:.4f}")

if p_val < 0.05:
    print("结论: 控制年龄和性别后，两组之间仍存在显著差异 (*p < 0.05)")
else:
    print("结论: 控制年龄和性别后，两组之间没有显著差异 (p >= 0.05)")
print("="*50)


# ==================== 绘制箱体图 ====================
plt.figure(figsize=(8, 6))

# 绘制箱线图
ax = sns.boxplot(x='Group', y=target_metric, data=df_all, width=0.4, palette='Set2')
# 叠加散点
sns.stripplot(x='Group', y=target_metric, data=df_all, color='black', alpha=0.3, jitter=True)

# 动态生成标题和坐标轴
plt.title(f'{target_metric} Scores (Adjusted for Age & Sex)\n{status_text} (p = {p_val:.4f})', fontsize=14)
plt.ylabel(f'{target_metric} Scores', fontsize=12)
plt.xlabel('', fontsize=12)

# 增加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# 保存文件
save_filename = f"{target_metric}_{'Cleaned' if REMOVE_OUTLIERS else 'Original'}_ANCOVA_Comparison.png"
plt.savefig(save_filename, dpi=300)
print(f"✅ 图表已保存至: {save_filename}")

plt.show()