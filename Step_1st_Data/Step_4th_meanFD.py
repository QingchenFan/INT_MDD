import glob
import pandas as pd
import os

# 你的原始路径
path = '/Volumes/ZLabData/BrainProject/brainproject_II/fmriprep/HC_DZ_1/sub-*V01/func/sub-*V01_task-rest_desc-confounds_timeseries.tsv'
#path = "/Volumes/ZLabData/BrainProject/brainproject_II/fmriprep/MDD_DZ_1/sub-*V01/func/sub-*V01_task-rest_desc-confounds_timeseries.tsv"
fl = glob.glob(path)

# 创建一个空列表，用于存储所有被试的计算结果
mean_fd_results = []

for i in fl:
    if not os.path.exists(i):
        continue

    if 'sub-07000043V01' in i:
        continue
    # 提取被试ID
    sub_id = i.split('/')[-3]
    print(sub_id)
    # 读取 tsv 文件
    conf = pd.read_csv(i, sep='\t')

    # 提取 framewise_displacement 列
    fd_series = conf['framewise_displacement']

    # 【核心修改点】：筛选出 <= 0.3 的值，排除 > 0.3 的高头动点
    # 注意：Pandas 会自动处理开头的 NaN 值，NaN 不会参与计算
    clean_fd_series = fd_series[fd_series <= 0.3]

    # 计算剔除坏点后的平均 FD
    mean_fd = clean_fd_series.mean()

    # 将被试ID和对应的平均FD添加到结果列表中
    mean_fd_results.append({
        'subject_id': sub_id,
        'mean_fd': mean_fd,  # 剔除坏点后的平均值
        #'total_volumes': len(fd_series),  # 顺便记录一下总时间点数 (可选)
        #'kept_volumes': len(clean_fd_series)  # 顺便记录一下保留了多少个时间点 (可选)
    })

# 转换为 DataFrame
df_results = pd.DataFrame(mean_fd_results)

# 打印前几行查看效果
print(df_results.head())

# 保存结果
df_results.to_csv('DZ_HC_subjects_mean_fd2.csv', index=False)