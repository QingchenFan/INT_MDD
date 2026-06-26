import pandas as pd
import numpy as np
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

'''
    HC MDD 8个网络进行比较 GMV
    - 统计：使用多元回归 (ANCOVA) 严格控制年龄、性别和 TIV，计算显著性
    - 绘图：计算残差以展示去混淆后的数据分布
'''
# ================== 1. 读取数据 ==================
hc_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step3_Structure/HC_GMV_7net_agesex.csv')
mdd_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step3_Structure/MDD_GMV_7Net_agesex.csv')

# 打标签并合并
hc_df['Group'] = 'HC'
mdd_df['Group'] = 'MDD'
combined_df = pd.concat([hc_df, mdd_df], axis=0).reset_index(drop=True)

# 定义变量 (添加 _GMV 后缀以精确匹配您的 CSV 列名)
networks = ['subcortical_GMV', 'Visual_GMV', 'Somatomotor_GMV', 'Dorsal_Attention_GMV',
            'Ventral_Attention_GMV', 'Limbic_GMV', 'Frontoparietal_GMV', 'Default_GMV']

# ***修改点 1：将 TIV 加入协变量列表***
covariates = ['age', 'sex', 'TIV']

print(f"总样本数: {len(combined_df)} (HC: {len(hc_df)}, MDD: {len(mdd_df)})")

# ================== 2. 严谨的统计检验 (多元回归 ANCOVA) ==================
results = []

for net in networks:
    # ***修改点 2：在回归公式中加入 TIV***
    # 建立多元回归模型: 网络GMV ~ 组别 + 年龄 + 性别 + TIV
    formula = f"{net} ~ C(Group, Treatment(reference='HC')) + age + C(sex) + TIV"

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

print("\n=== 多元回归控制 Age/Sex/TIV 后的组间差异 (ANCOVA) ===")
print(result_df)

# ================== 3. 画图数据准备 (仅回归年龄、性别、TIV求残差) ==================
# 这部分仅为了生成干净的、去除了协变量影响的散点用于可视化
corrected_data = combined_df.copy()

for net in networks:
    mask = combined_df[[net] + covariates].notnull().all(axis=1)
    temp_df = combined_df[mask]

    X = temp_df[covariates]
    X = sm.add_constant(X)
    y = temp_df[net]

    resid_model = sm.OLS(y, X).fit()
    # 将残差存回原数据框，加上原始均值保持量级
    corrected_data.loc[mask, net] = resid_model.resid + y.mean()

# ================== 4. 绘图 (使用残差数据 + 多元回归的显著性P值) ==================
plt.figure(figsize=(16, 10))

# 转换数据格式以适配 seaborn (将变量名修改为 GMV)
plot_data = corrected_data.melt(id_vars='Group', value_vars=networks, var_name='Network', value_name='GMV')

# 美化：去除 X 轴标签里的 '_GMV' 后缀，让画出的图表更干净
plot_data['Network'] = plot_data['Network'].str.replace('_GMV', '')

# 绘制箱体图
ax = sns.boxplot(x='Network', y='GMV', hue='Group', data=plot_data,
                 palette={'HC': '#1f77b4', 'MDD': '#d62728'}, width=0.6, showfliers=False)

# 绘制散点图
sns.stripplot(x='Network', y='GMV', hue='Group', data=plot_data,
              dodge=True, alpha=0.4, jitter=True, size=3, palette={'HC': '#1f77b4', 'MDD': '#d62728'})

# 标注显著性 (根据第二步严谨算出的 result_df)
for i, net in enumerate(networks):
    # 根据网络名称找到对应的校正后 P 值
    p = result_df.loc[result_df['Network'] == net, 'p_fdr'].values[0]
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

    # 找到最大值用于确定文字高度
    y_max = corrected_data[net].max()
    plt.text(i, y_max * 1.02, sig, ha='center', fontsize=14, fontweight='bold')

# 修复重复图例的问题 (boxplot 和 stripplot 都会生成图例)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:2], labels[0:2], title='Group', loc='upper right')

# ***修改点 3：更新图表标题和轴标签***
plt.title('Comparison across Networks (Adjusted for Age, Sex, and TIV)', fontsize=16)
plt.ylabel('Adjusted GMV (Residuals + Mean)', fontsize=14)
plt.xticks(rotation=45)

# 为了防止星星被图表顶部边缘裁掉，稍微拉高 Y 轴上限
bottom, top = plt.ylim()
plt.ylim(bottom, top + (top - bottom) * 0.05)

sns.despine()
plt.tight_layout()
plt.savefig('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step3_Structure/HCMDD_8NetGMV_Diff.png', dpi=300)