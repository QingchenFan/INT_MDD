import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import statsmodels.stats.multitest as smm
data = pd.read_csv("/Volumes/QC/INT/INT_BN246_HC135BP_allMDD/Results/allDataMDD_HAMD_final.csv")

hamd = np.array(data['HAMD17'])

brainRegion = data.columns.tolist()
del brainRegion[:2]
rvalue = []
pvalue = []
for i in brainRegion:
    x = np.array(data[i])
    y = hamd

    #corr, p_value = pearsonr(x, y)
    corr, p_value = spearmanr(x, y)
    rvalue.append(corr)
    pvalue.append(p_value)
    print(i)
    print(f"Spearman相关系数: {corr}")
    print(f"p值: {p_value}")

# 对p值进行FDR校正
rejected, fdr_pvalue, _, _ = smm.multipletests(pvalue, method='fdr_bh')

# 创建DataFrame，将rvalue、pvalue、fdr_pvalue对应保存
result_df = pd.DataFrame({
    'rvalue': rvalue,
    'pvalue': pvalue,
    'fdr-pvalue': fdr_pvalue
})

# 将结果保存到CSV文件中，可根据实际需求修改文件路径及文件名
result_df.to_csv('./results.csv', index=False)