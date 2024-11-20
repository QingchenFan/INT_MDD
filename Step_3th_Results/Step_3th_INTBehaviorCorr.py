import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import statsmodels.stats.multitest as smm
data = pd.read_csv("/Volumes/QC/INT/INT_BN246_HC135BP_allMDD/Results/Data135DLMDD_PHQ9_final.csv")

behscore = np.array(data['PHQ9'])

brainRegion = data.columns.tolist()
del brainRegion[:2]
srvalue = []
spvalue = []
rvalue = []
pvalue = []
for i in brainRegion:
    x = np.array(data[i])
    y = behscore

    corr, p_value = pearsonr(x, y)
    rvalue.append(corr)
    pvalue.append(p_value)

    scorr, sp_value = spearmanr(x, y)
    srvalue.append(scorr)
    spvalue.append(sp_value)

    print(f"Spearman相关系数: {scorr}", f"p值: {sp_value}")
    print(f"Pearson相关系数: {corr}", f"p值: {p_value}")

# 对p值进行FDR校正
rejected, fdr_pvalue, _, _ = smm.multipletests(pvalue, method='fdr_bh')
srejected, sfdr_pvalue, _, _ = smm.multipletests(spvalue, method='fdr_bh')

# 创建DataFrame，将rvalue、pvalue、fdr_pvalue对应保存
result_df = pd.DataFrame({
    'rvalue-s': srvalue,
    'pvalue-s': spvalue,
    'fdr-pvalue-s': sfdr_pvalue,
    'rvalue': rvalue,
    'pvalue': pvalue,
    'fdr-pvalue': fdr_pvalue,

})

# 将结果保存到CSV文件中，可根据实际需求修改文件路径及文件名
result_df.to_csv('./PHQ9_results.csv', index=False)