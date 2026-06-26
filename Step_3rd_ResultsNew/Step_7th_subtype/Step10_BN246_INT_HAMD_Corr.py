import pandas as pd
import numpy as np
from scipy.stats import pearsonr  # 若需非参数检验，可导入 spearmanr
from statsmodels.stats.multitest import multipletests

# 1. 读取数据
df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step10_BN246_INT_HAMD_Correlation/subtype2_BN246_INT_HAMD.csv')

# 2. 提取脑区列名 (排除非脑区数据 'subID' 和 'HAMD_0w')
brain_regions = [col for col in df.columns if col not in ['subID', 'HAMD_0w']]

# 3. 循环计算 HAMD_0w 与每个脑区 INT 的相关性
results = []
for region in brain_regions:
    # 提取这两列并删除可能存在的缺失值 (NaN)
    valid_data = df[['HAMD_0w', region]].dropna()

    # 计算皮尔逊相关系数 (r) 和 p 值
    r, p = pearsonr(valid_data['HAMD_0w'], valid_data[region])

    # 将结果保存到字典列表中
    results.append({
        'Brain_Region': region,
        'r_value': r,
        'p_value': p
    })

# 转换为 DataFrame
results_df = pd.DataFrame(results)

# 4. 进行 p 值多重比较校正 (使用 FDR-BH 方法)
# alpha=0.05 表示校正后的显著性水平
reject, pvals_corrected, _, _ = multipletests(results_df['p_value'], alpha=0.05, method='fdr_bh')

# 将校正后的结果加入 DataFrame
results_df['p_value_fdr'] = pvals_corrected
results_df['significant'] = reject  # True 表示在设定 alpha 下显著

# 5. 按照原始 p 值从小到大排序
results_df = results_df.sort_values(by='p_value')

# 6. 保存结果到新的 CSV 文件
results_df.to_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step10_BN246_INT_HAMD_Correlation/'
                  's2_HAMD_INT_correlation_results.csv', index=False)

# 打印前 10 个最相关的脑区
print(results_df.head(10))