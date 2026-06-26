import pandas as pd
'''
    计算最大值和最小值的差
'''
# 1. 读取原始数据
df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step2_Ttest/HC_INT20_agesex_FD.csv')

# 2. 识别脑区列 (排除掉基本信息列)
basic_cols = ['subID', 'mean_fd', 'age', 'sex']
brain_regions = [col for col in df.columns if col not in basic_cols]

# 3. 计算每个被试在 246 个脑区上的最大值、最小值以及差值
df['max_region'] = df[brain_regions].max(axis=1)
df['min_region'] = df[brain_regions].min(axis=1)
df['difference'] = df['max_region'] - df['min_region']

# 4. 提取所需要的列
result_df = df[['subID', 'age', 'sex', 'mean_fd', 'max_region', 'min_region', 'difference']]

# 5. 保存结果到新的 CSV 文件
output_filename = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_INTRange/Step1_BN246_INTRange/' \
                  'HC_BN246INT_Range.csv'
result_df.to_csv(output_filename, index=False)

print(f"处理完成并已保存！前 5 行数据如下：\n{result_df.head()}")