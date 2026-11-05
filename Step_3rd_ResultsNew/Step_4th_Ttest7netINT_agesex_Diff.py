import pandas as pd
import numpy as np
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

'''
    HC MDD 8个网络进行比较 INT 
    - 统计：使用多元回归 (ANCOVA) 严格控制年龄、性别和头动(mean_fd)，计算显著性
    - 绘图：计算残差以展示去混淆后的数据分布
'''
# ================== 1. 读取数据 ==================
# 确保你的 CSV 文件中包含 'mean_fd' 这一列
hc_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_Match/HC_INT20_7net_agesex_FD.csv')
mdd_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_Match/MDD_INT20_7net_agesex.csv')

# 打标签并合并
hc_df['Group'] = 'HC'
mdd_df['Group'] = 'MDD'
combined_df = pd.concat([hc_df, mdd_df], axis=0).reset_index(drop=True)

# 定义变量
networks = ['subcortical', 'Visual', 'Somatomotor', 'Dorsal_Attention',
            'Ventral_Attention', 'Limbic', 'Frontoparietal', 'Default']
# 【修改点 1】将 mean_fd 加入协变量列表
covariates = ['age', 'sex']

print(f"总样本数: {len(combined_df)} (HC: {len(hc_df)}, MDD: {len(mdd_df)})")

# ================== 2. 严谨的统计检验 (多元回归 ANCOVA) ==================
results = []

for net in networks:
    # 【修改点 2】修改回归公式，将 mean_fd 作为连续变量加入模型
    formula = f"{net} ~ C(Group, Treatment(reference='HC')) + age + C(sex) "

    # 拟合 OLS 模型
    model = smf.ols(formula, data=combined_df).fit()

    # 提取 MDD 组相较于 HC 组的主效应系数
    coef_name = "C(Group, Treatment(reference='HC'))[T.MDD]"
    t_stat = model.tvalues[coef_name]
    p_val = model.pvalues[coef_name]

    # 计算原始均值用于表格展示
    hc_raw_mean = combined_df[combined_df['Group'] == 'HC'][net].mean()
    mdd_raw_mean = combined_df[combined_df['Group'] == 'MDD'][net].mean()

    results.append({
        'Network': net,
        'HC_mean_raw': round(hc_raw_mean, 4),
        'MDD_mean_raw': round(mdd_raw_mean, 4),
        't': round(t_stat, 3),
        'p_raw': p_val
    })

# 转换为 DataFrame 并进行 FDR 校正
result_df = pd.DataFrame(results)
_, p_fdr = fdrcorrection(result_df['p_raw'], alpha=0.05)
result_df['p_fdr'] = p_fdr
result_df['significant'] = result_df['p_fdr'] < 0.05

# 排序方便查看
result_df = result_df.sort_values('p_fdr')

print("\n=== 多元回归控制 Age/Sex/Mean_FD 后的组间差异 (ANCOVA) ===")
print(result_df)

# ================== 3. 画图数据准备 (回归年龄、性别、头动求残差) ==================
# 这部分仅为了生成干净的、去除了所有协变量（含头动）影响的散点用于可视化
corrected_data = combined_df.copy()

for net in networks:
    # 【核心修改点 3】这里因为使用 sm.OLS，需要手动处理分类变量 sex
    # 我们可以通过 pd.get_dummies 将 sex 转换为哑变量，或者直接使用包含 mean_fd 的设计矩阵
    mask = combined_df[[net] + covariates].notnull().all(axis=1)
    temp_df = combined_df[mask]

    # 为了保持与前面 smf.ols 行为一致（处理性别分类变量），这里改用更加稳妥的 smf.ols 来提取残差
    # 这样可以自动帮你把 C(sex) 转成 0/1 编码，避免手动加 constant 和处理离散变量的麻烦
    resid_formula = f"{net} ~ age + C(sex) "
    resid_model = smf.ols(resid_formula, data=temp_df).fit()

    # 将残差存回原数据框，加上原始均值保持量级
    corrected_data.loc[mask, net] = resid_model.resid + temp_df[net].mean()

# ================== 4. 绘图 (使用残差数据 + 多元回归的显著性P值) ==================
plt.figure(figsize=(16, 10))

# 转换数据格式以适配 seaborn
plot_data = corrected_data.melt(id_vars='Group', value_vars=networks, var_name='Network', value_name='INT')

# 绘制箱体图
ax = sns.boxplot(x='Network', y='INT', hue='Group', data=plot_data,
                 palette={'HC': '#1f77b4', 'MDD': '#d62728'}, width=0.6, showfliers=False)

# 绘制散点图
sns.stripplot(x='Network', y='INT', hue='Group', data=plot_data,
              dodge=True, alpha=0.4, jitter=True, size=3, palette={'HC': '#1f77b4', 'MDD': '#d62728'}, legend=False)

# 标注显著性 (根据第二步严谨算出的 result_df)
for i, net in enumerate(networks):
    # 根据网络名称找到对应的校正后 P 值
    p = result_df.loc[result_df['Network'] == net, 'p_fdr'].values[0]
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.06 else 'n.s.'

    # 找到最大值用于确定文字高度
    y_max = corrected_data[net].max()
    plt.text(i, y_max * 1.02, sig, ha='center', fontsize=14, fontweight='bold')

# 修复重复图例的问题
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:2], labels[0:2], title='Group', loc='upper right')

plt.title('Comparison across Networks (Adjusted for Age, Sex, and Mean FD)', fontsize=16)
plt.ylabel('Adjusted INT (Residuals + Mean)', fontsize=14)
plt.xticks(rotation=45)

# 为了防止星星被图表顶部边缘裁掉，稍微拉高 Y 轴上限
bottom, top = plt.ylim()
plt.ylim(bottom, top + (top - bottom) * 0.05)

sns.despine()
plt.tight_layout()
plt.savefig('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_Match/Step1_HC_MDD_Diff/HCMDD_8NetINT_Diff.png', dpi=300)
plt.show()