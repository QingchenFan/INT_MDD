import pandas as pd

# 1. 读取原始数据文件
# 请根据您本地的实际路径修改文件路径，例如: '/Volumes/QC/INT/.../subtype1_INT_7net_agesex_FD.csv'
file_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step2_Ttest/HC_INT20_7net_agesex_FD.csv'
df = pd.read_csv(file_path)

# 2. 定义用于计算的 8 个脑网络列名
network_cols = ['subcortical', 'Visual', 'Somatomotor', 'Dorsal_Attention',
                'Ventral_Attention', 'Limbic', 'Frontoparietal', 'Default']

# 3. 计算每个被试（每一行）在 8 个网络上的最大值、最小值，并得出差值
df['max_net'] = df[network_cols].max(axis=1)
df['min_net'] = df[network_cols].min(axis=1)
df['difference'] = df['max_net'] - df['min_net']

# 4. 提取需要保存的列：原数据中的基本信息和新计算的结果
keep_cols = ['subID', 'age', 'sex', 'mean_fd', 'max_net', 'min_net', 'difference']

# 过滤存在于当前 DataFrame 中的列，确保不出错
actual_keep_cols = [col for col in keep_cols if col in df.columns]

# 生成最终的数据表
result_df = df[actual_keep_cols]

# 5. 保存结果到新的 CSV 文件中
output_filename = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_INTRange/Step2_8Net_INTRange/' \
                  'HC_net_range.csv'
result_df.to_csv(output_filename, index=False)

print(f"数据已成功保存至: {output_filename}")
print(result_df.head())