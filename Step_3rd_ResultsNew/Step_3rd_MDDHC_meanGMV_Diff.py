import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# 1. 加载数据 (修改为 GrayVol 数据)
# 请根据您的实际路径进行调整
hc_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step3_Structure/HC_GrayVol246.csv')
mdd_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step3_Structure/MDD_GrayVol246.csv')

# 2. 添加组别标签
hc_df['Group'] = 'HC'
mdd_df['Group'] = 'MDD'

# 3. 合并数据集
combined_df = pd.concat([hc_df, mdd_df], ignore_index=True)

# 4. 识别脑区列并计算全脑灰质体积均值
# 这里的 covariates 增加了 'TIV'，防止它被当成脑区去计算均值
covariates = ['subID', 'TIV', 'age', 'sex', 'Group']
brain_regions = [col for col in combined_df.columns if col not in covariates]

# 计算 246 个脑区灰质体积的均值
combined_df['Global_GrayVol_Mean'] = combined_df[brain_regions].mean(axis=1)

# 5. 计算统计学差异 (控制年龄、性别和 TIV)
# 明确指定 HC 为参考组，确保 t 值的正负反映的是 MDD 相比 HC 的变化
formula = 'Global_GrayVol_Mean ~ C(Group, Treatment(reference="HC")) + age + C(sex) + TIV'
model = smf.ols(formula, data=combined_df).fit()

# 提取 MDD vs HC 的 t 值和 p 值
coef_name = 'C(Group, Treatment(reference="HC"))[T.MDD]'
t_val = model.tvalues[coef_name]
p_val = model.pvalues[coef_name]

# 6. 开始绘图
plt.figure(figsize=(8, 6))

# 绘制箱体图
sns.boxplot(
    x='Group',
    y='Global_GrayVol_Mean',
    data=combined_df,
    width=0.4,
    palette='Set2',
    showfliers=False,
    boxprops={'alpha': 0.6}
)

# 叠加散点图
sns.stripplot(
    x='Group',
    y='Global_GrayVol_Mean',
    data=combined_df,
    jitter=True,
    marker='o',
    alpha=0.6,
    color='black',
    size=5
)

# 7. 添加显著性标识 (如果在 p < 0.05 范围内才显示)
if p_val < 0.05:
    # 确定星号数量
    if p_val < 0.001:
        stars = '***'
    elif p_val < 0.01:
        stars = '**'
    else:
        stars = '*'

    # 获取数据 y 轴范围，用于动态计算横线和文本的位置
    y_max = combined_df['Global_GrayVol_Mean'].max()
    y_range = combined_df['Global_GrayVol_Mean'].max() - combined_df['Global_GrayVol_Mean'].min()

    # 设定连线的高度位置
    y_line = y_max + (y_range * 0.05)
    h = y_range * 0.02  # 竖向下沉的小刻度高度

    # Seaborn 中类别 x 轴的坐标默认是 0 (左侧组) 和 1 (右侧组)
    x1, x2 = 0, 1

    # 画显著性连线: |---|
    plt.plot([x1, x1, x2, x2], [y_line, y_line + h, y_line + h, y_line], lw=1.5, c='black')

    # 在连线上方居中写上星号和 t, p 值
    annot_text = f"{stars}\nt = {t_val:.2f}, p = {p_val:.3f}"
    plt.text((x1 + x2) * 0.5, y_line + h, annot_text, ha='center', va='bottom', color='black', fontsize=11)

# 8. 美化和标签设置
# 增加 pad=25 防止标题和上面的显著性 P 值文本重叠
plt.title('Global Gray Matter Volume Mean: HC vs MDD', fontsize=14, pad=25)
plt.ylabel('Global Gray Matter Volume Mean', fontsize=12)
plt.xlabel('Group', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 扩大 y 轴显示上限，防止显著性的标识和文字被截断
bottom, top = plt.ylim()
# 如果有星号标注，需要留出更多空间，这里预留 15% 顶部空间
if p_val < 0.05:
    plt.ylim(bottom, top + (top - bottom) * 0.15)
else:
    plt.ylim(bottom, top + (top - bottom) * 0.05)

plt.tight_layout()

# 9. 显示或保存图片
output_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step3_Structure/HCMDD_mean246GrayVol.png'
plt.savefig(output_path, dpi=300)
print(f"绘图完成！图片已保存至: {output_path}")
print(f"统计结果 (控制 Age, Sex, TIV): t={t_val:.3f}, p={p_val:.4f}")