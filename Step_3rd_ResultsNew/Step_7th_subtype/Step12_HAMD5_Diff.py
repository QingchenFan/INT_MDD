import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据 (保留您的绝对路径)
df1 = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step12_HAMD5_Diff/subtype1_HAMD5_zero.csv')
df2 = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step12_HAMD5_Diff/subtype2_HAMD5_zero.csv')

# 添加亚型标签列，以便后续合并和绘图
df1['Subtype'] = 'Subtype 1'
df2['Subtype'] = 'Subtype 2'

# 合并两个数据集
df_combined = pd.concat([df1, df2])

# 需要比较的维度列表
dimensions = ['Dimension_1', 'Dimension_2', 'Dimension_3', 'Dimension_4', 'Dimension_5']

# 2. 批量进行双样本 t 检验 (Welch's t-test, 假设方差不齐)
results = []
for dim in dimensions:
    # equal_var=False 表示使用 Welch's t-test
    t_stat, p_val = stats.ttest_ind(df1[dim], df2[dim], equal_var=False)
    results.append({
        'Dimension': dim,
        't-statistic': t_stat,
        'p-value': p_val
    })

# 打印检验结果
results_df = pd.DataFrame(results)
print("双样本 t 检验结果:")
print(results_df)  # 补全：实际打印出 DataFrame 表格

# 3. 准备绘图数据 (补全：使用 melt 函数将宽表转换为长表，适应 seaborn 绘图格式)
df_melted = pd.melt(df_combined,
                    id_vars=['Subtype'],
                    value_vars=dimensions,
                    var_name='Dimension',
                    value_name='Score')

# 4. 绘制共用坐标轴的箱线图 (补全：实际绘图代码)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_melted, x='Dimension', y='Score', hue='Subtype')

# 设置图表标题和标签
plt.title('Comparison of Dimensions between Subtypes')
plt.xlabel('Dimensions')
plt.ylabel('Scores')
plt.legend(title='Subtype')
plt.tight_layout()

# 5. 保存图片到指定路径
# 添加了 dpi=300 可以让保存的图片更清晰
plt.savefig('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step12_HAMD5_Diff/HAMD5_Diff_zero.png', dpi=300)

# 如果您想在运行代码时也弹窗预览图片，可以取消下面这行的注释
# plt.show()