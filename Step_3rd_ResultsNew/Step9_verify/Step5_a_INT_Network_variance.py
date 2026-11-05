import pandas as pd

# 1. 读取原始数据文件
# 请根据您本地的实际路径修改文件路径
file_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_verify/INT_covariate/HC_DZ_INT_8net.csv'
df = pd.read_csv(file_path)

# 2. 识别基本信息列和脑区列
# 将不需要计算方差的基础信息列排除
basic_cols = ['subID', 'mean_fd', 'age', 'sex', 'mean_fd']
brain_regions = [col for col in df.columns if col not in basic_cols]

# 3. 计算每个被试（每一行）在 246 个脑区上的方差
df['variance'] = df[brain_regions].var(axis=1)

# 4. 提取需要保存的列：原数据中的基本信息和新计算的方差结果
keep_cols = ['subID', 'age', 'sex', 'mean_fd',  'variance']

# 过滤存在于当前 DataFrame 中的列，确保不出错
actual_keep_cols = [col for col in keep_cols if col in df.columns]

# 生成最终的数据表
result_df = df[actual_keep_cols]

# 5. 保存结果到新的 CSV 文件中
output_filename = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_verify/INT_covariate/Step4_Variance_Diff/HC_INT_7net_variance.csv'
result_df.to_csv(output_filename, index=False)

print(f"数据已成功保存至: {output_filename}")
print(result_df.head())