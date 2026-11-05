import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# 1. 加载数据
hc_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_Match/HC_INT20_agesex_FD.csv')
mdd_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_Match/MDD_INT20_agesex_FD.csv')

# 2. 添加组别标签
hc_df['Group'] = 'HC'
mdd_df['Group'] = 'MDD'

# 3. 合并数据集
combined_df = pd.concat([hc_df, mdd_df], ignore_index=True)

# 4. 识别协变量与脑区列 (在此处加入 'mean_fd')
# 确保 mean_fd 不会被混入脑区均值计算中
covariates = ['subID', 'age', 'sex', 'Group','mean_fd']
brain_regions = [col for col in combined_df.columns if col not in covariates]

# 5. 计算全脑均值
# 警告：由于两组数据的 brain_regions 不同，计算的 Global_Brain_Mean 可能不具有直接可比性
combined_df['Global_Brain_Mean'] = combined_df[brain_regions].mean(axis=1)

# 6. 计算统计学差异 (控制年龄、性别和 mean_fd)
# 修正后的公式加入了 + mean_fd
formula = 'Global_Brain_Mean ~ C(Group) + age + C(sex) + mean_fd'
model = smf.ols(formula, data=combined_df).fit()

# 提取 MDD vs HC 的 t 值和 p 值
t_val = model.tvalues['C(Group)[T.MDD]']
p_val = model.pvalues['C(Group)[T.MDD]']

# 7. 开始绘图
plt.figure(figsize=(8, 6))

# 绘制箱体图
sns.boxplot(
    x='Group',
    y='Global_Brain_Mean',
    data=combined_df,
    width=0.4,
    palette='Set2',
    showfliers=False,
    boxprops={'alpha': 0.6}
)

# 叠加散点图
sns.stripplot(
    x='Group',
    y='Global_Brain_Mean',
    data=combined_df,
    jitter=True,
    marker='o',
    alpha=0.6,
    color='black',
    size=5
)

# 8. 添加显著性标识
if p_val < 0.05:
    if p_val < 0.001:
        stars = '***'
    elif p_val < 0.01:
        stars = '**'
    else:
        stars = '*'

    y_max = combined_df['Global_Brain_Mean'].max()
    y_range = combined_df['Global_Brain_Mean'].max() - combined_df['Global_Brain_Mean'].min()
    y_line = y_max + (y_range * 0.05)
    h = y_range * 0.02
    x1, x2 = 0, 1

    plt.plot([x1, x1, x2, x2], [y_line, y_line + h, y_line + h, y_line], lw=1.5, c='black')
    annot_text = f"{stars}\nt = {t_val:.2f}, p = {p_val:.3f}"
    plt.text((x1 + x2) * 0.5, y_line + h, annot_text, ha='center', va='bottom', color='black', fontsize=11)

# 9. 美化和标签设置
plt.title('Global Brain Mean Comparison (Controlled for FD)', fontsize=14, pad=25)
plt.ylabel('Global Brain Mean', fontsize=12)
plt.xlabel('Group', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

bottom, top = plt.ylim()
plt.ylim(bottom, top + (top - bottom) * 0.15)

plt.tight_layout()


# 9. 显示或保存图片
plt.savefig('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_Match/Step1_HC_MDD_Diff/HCMDD_mean246INT.png', dpi=300)
print(f"绘图完成！图片已保存。统计结果: t={t_val:.3f}, p={p_val:.4f}")