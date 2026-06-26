import pandas as pd
'''
    计算均值
'''
# 1. 加载原始数据
# 假设文件名为 'subtype1_INT.csv'
file_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step2_Ttest/HC_INT20_agesex.csv'
df = pd.read_csv(file_path)

# 2. 提取脑区列（自动排除 'subID' 列）
region_cols = [col for col in df.columns if col != 'subID']



# 3. 计算每个脑区的均值（按列计算所有被试的平均值）
mean_values = df[region_cols].mean()

# 4. 构造输出数据框
# 将 Series 转换为 DataFrame 并转置 (.T)，使脑区名成为表头，均值成为第一行数据
mean_df = pd.DataFrame(mean_values).T

# 5. 保存到 CSV
output_csv = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step2_BN246mean_INT/HC_brain_regions_mean_INT.csv'
# index=False 表示不保存行索引（即不保存 0 这个数字）
mean_df.to_csv(output_csv, index=False)

