import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
'''
此代码计算共变网络未考虑年龄 性别  FD --不考虑用此代码
'''
# 1. 读取数据
df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_Covariation/subtype2_GMV246.csv')

# 2. 数据预处理
# 将 'subID' 设置为索引，排除在共变网络计算之外
if 'subID' in df.columns:
    df = df.set_index('subID')

# 3. 计算共变网络（皮尔逊相关系数矩阵）
# 这里计算的是列与列（即各个脑区）之间的相关性
corr_matrix = df.corr()

# 4. 导出计算结果
corr_matrix.to_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_Covariation/subtype2_GMVcovariance_network.csv')
print(f"数据维度 (受试者数量, 脑区数量): {df.shape}")
print(f"共变网络矩阵维度: {corr_matrix.shape}")

# 5. 绘制网络热力图可视化
plt.figure(figsize=(12, 10))
# 使用 coolwarm 颜色映射，正相关偏红，负相关偏蓝
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True,
            xticklabels=False, yticklabels=False)
plt.title('Covariance Network (Correlation Matrix)')
plt.tight_layout()
plt.savefig('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_Covariation/subtype2_GMVcovariance_network_heatmap.png')