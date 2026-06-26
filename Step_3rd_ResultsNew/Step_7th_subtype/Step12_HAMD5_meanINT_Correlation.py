import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据 (您可以替换为您本机的绝对路径)
df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step13_HAMD5_meanINT_Correlation/subtype2_meanINT_HAMD5.csv')

# 2. 定义变量
target = 'mean_INT'
dimensions = ['Dimension_1', 'Dimension_2', 'Dimension_3', 'Dimension_4', 'Dimension_5']

# 3. 计算相关性 (Pearson 相关系数)
results = []
for dim in dimensions:
    # pearsonr 输入顺序不影响结果，这里保持一致性
    r, p_val = stats.pearsonr(df[dim], df[target])
    results.append({
        'Dimension': dim,
        'Pearson_r': r,
        'p-value': p_val
    })

results_df = pd.DataFrame(results)
print("相关性分析结果:")
print(results_df)

# 4. 绘制散点图
# 【修改1】由于 x 轴和 y 轴互换，现在5个图共用的是 y 轴 (mean_INT)，所以改为 sharey=True
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)

# 循环遍历维度，并在对应的子图 (ax) 上绘图
for i, dim in enumerate(dimensions):
    # 【修改2】将 x 和 y 的参数值互换：x=dim, y=target
    sns.regplot(data=df, x=dim, y=target, ax=axes[i], scatter_kws={'alpha': 0.6})

    # 设置子图标题，在标题加上 r 值和 p 值
    r_val = results_df.loc[i, 'Pearson_r']
    p_val = results_df.loc[i, 'p-value']
    axes[i].set_title(f'{dim}\nr={r_val:.3f}, p={p_val:.3f}')

    # 【修改3】动态设置每个图的 x 轴标签
    axes[i].set_xlabel('Score')

# 【修改4】只需要在最左边第一个子图上设置 y 轴标签即可
axes[0].set_ylabel('mean_INT')

# 自动调整子图间距
plt.tight_layout()

# 保存或者展示图片
plt.savefig('//Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step13_HAMD5_meanINT_Correlation/s2_meanINT_HAMD5.png', dpi=300)
plt.show()