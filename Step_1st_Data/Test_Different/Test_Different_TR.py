import pandas as pd
from scipy import stats
# 加载第一个数据集
df1 = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC_MDD/INT_Results/Test_Different/INTvalue_ADDZ.csv')

# 加载第二个数据集
df2 = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC_MDD/INT_Results/Test_Different/INTvalue_HCPHX.csv')


# 初始化一个字典来存储每个脑区的 p 值和检验类型
results = {}

# 遍历每一个脑区（除了 subID 列）
for col in df1.columns[1:]:
    # 对每个脑区的数据进行 Shapiro - Wilk 正态性检验
    sw1 = stats.shapiro(df1[col])
    sw2 = stats.shapiro(df2[col])

    # 如果两个数据集的该脑区数据都通过正态性检验（p 值大于 0.05），则进行独立样本 t 检验
    if sw1.pvalue > 0.05 and sw2.pvalue > 0.05:
        test_result = stats.ttest_ind(df1[col], df2[col])
        test_type = 't - test'
    else:
        # 否则进行 Mann - Whitney U 检验
        test_result = stats.mannwhitneyu(df1[col], df2[col])
        test_type = 'Mann - Whitney U test'

    # 存储结果
    results[col] = {
        'p_value': test_result.pvalue,
        'test_type': test_type
    }

# 将结果转换为 DataFrame 以便于查看
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.to_csv('./results.csv', index=True)
# 查看有显著差异（p 值小于 0.05）的脑区数量
significant_count = (results_df['p_value'] < 0.05).sum()

# 查看结果
print('有显著差异的脑区数量:', significant_count, '各脑区的检验结果:', results_df)