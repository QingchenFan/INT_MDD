import pandas as pd

# 1. 读取原始数据
df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step2_Ttest/HC_INT20_7net_agesex.csv')

# 2. 定义 8 个网络的列名
networks = ['subcortical', 'Visual', 'Somatomotor', 'Dorsal_Attention',
            'Ventral_Attention', 'Limbic', 'Frontoparietal', 'Default']

# 3. 计算每个被试（行维度，axis=1）在 8 个网络连接值上的方差
df['network_variance'] = df[networks].var(axis=1)

# 4. 提取需要保留的列：subID, age, sex 以及新计算的方差
result_df = df[['subID', 'age', 'sex', 'network_variance']]

# 5. 保存到新的 CSV 文件中，index=False 表示不保存行索引
output_filename = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_NetworkHierarchy/HC_variances.csv'
result_df.to_csv(output_filename, index=False)

print("处理完成并已保存！前5行数据如下：")
print(result_df.head())