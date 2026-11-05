import glob
import scipy.io
import pandas as pd
import numpy as np
import os
from statsmodels.stats.multitest import multipletests

# 1. 设置路径
resPath = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_verify/INT_covariate/subtype_permutation/Results_All/perm_result_*/*.mat"
mat_paths = glob.glob(resPath)
print(f"找到{len(mat_paths)}个置换结果文件")

# 提取每个文件对应的 perm_result_* 文件夹名作为标识
perm_folder_names = []
for mat_file in mat_paths:
    # 获取mat文件所在文件夹完整路径
    folder_path = os.path.dirname(mat_file)
    # 截取末尾文件夹名 perm_result_xxx
    folder_name = os.path.basename(folder_path)
    perm_folder_names.append(folder_name)

# 2. 读取真实值
real_data = scipy.io.loadmat('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_verify/INT_covariate/HYDRA_results.mat')
real_ari = real_data['ARI'].flatten() # 真实值: [ari_k1, ari_k2, ari_k3, ari_k4, ari_k5]

# 3. 循环读取所有置换结果
all_ari_list = []
for i in mat_paths:
    mat = scipy.io.loadmat(i)
    all_ari_list.append(mat['ARI'].flatten())

ari_matrix = np.array(all_ari_list)

# 4. 创建并保存 ARI 结果 CSV
# 第一列 ID 改为 perm_result_xxx 文件夹名称
df = pd.DataFrame(ari_matrix, columns=['k=1', 'k=2', 'k=3', 'k=4', 'k=5'])
df.insert(0, 'ID', perm_folder_names)
df.to_csv('permutation_ari_results_verify.csv', index=False)

# 5. 计算原始 p 值 (置换检验p值，+1校正避免p=0)
# 比较每一列（即每一个 k）的置换结果与对应的真实值
count_greater = np.sum(ari_matrix >= real_ari, axis=0)
p_values = (count_greater + 1) / (len(mat_paths) + 1)

# 6. 多重检验p值校正
# 方法1: Bonferroni 严格校正（控制FWER）
# 方法2: FDR-BH 错误发现率校正（常用、更宽松，科研主流）
# is_sig: 校正后是否显著(alpha=0.05)
alpha = 0.05

# Bonferroni校正
bonferroni_corrected = multipletests(p_values, alpha=alpha, method='bonferroni')
bonferroni_p = bonferroni_corrected[1]
bonferroni_sig = bonferroni_corrected[0]

# FDR-BH校正
bh_corrected = multipletests(p_values, alpha=alpha, method='fdr_bh')
bh_p = bh_corrected[1]
bh_sig = bh_corrected[0]

# 7. 保存包含校正p值的完整结果
results_p = pd.DataFrame({
    'k': [1, 2, 3, 4, 5],
    'Real_ARI': real_ari,
    'raw_p_value': p_values,
    'bonferroni_p': bonferroni_p,
    'bonferroni_significant': bonferroni_sig,
    'fdr_bh_p': bh_p,
    'fdr_bh_significant': bh_sig
})
results_p.to_csv('permutation_p_values_corrected_verify.csv', index=False)

# 8. 打印结果
print("="*60)
print("处理完成！结果汇总：")
print(f"置换ARI表格形状: {df.shape}")
print(f"显著性检验阈值 alpha = {alpha}")
print("="*60)
print(results_p.round(4))
print("="*60)
print("文件输出：")
print("1. permutation_ari_results.csv - 所有置换ARI原始结果")
print("2. permutation_p_values_corrected.csv - 原始p值+校正p值+显著性结果")
