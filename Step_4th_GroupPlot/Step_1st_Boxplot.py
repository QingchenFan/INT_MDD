import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 从 CSV 文件中读取数据
df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_BP135MDD/Results/INTvalue_HCgroup_net7_test.csv')

# 将数据转换为长格式
df_long = df.melt(id_vars=['subID', '0'],
                 value_vars=['Visual', 'Somatomotor', 'Dorsal Attention',
                            'Ventral Attention', 'Limbic', 'Frontoparietal', 'Default'],
                 var_name='Network',
                 value_name='Value')

# 计算每个网络的z轴值（这里假设是均值，可根据实际需求修改）
network_z_values = df_long.groupby('Network')['Value'].mean().sort_values(ascending=False)

# 获取排序后的网络顺序
network_order = network_z_values.index.tolist()

plt.figure(figsize=(10, 8))

# 绘制横向箱体图，使用排序后的网络顺序
ax = sns.boxplot(x='Value',
            y='Network',
            data=df_long,
            order=network_order,
            orient='h',
            palette='vlag',
            width=0.7)

# 设置背景颜色为透明
ax.set_facecolor('none')

# 关闭网格线显示
ax.grid(False)

# 确保坐标轴可见
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置坐标轴颜色和线宽
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

# 添加标题和标签
plt.title('Distribution of Network Values across Subjects (Sorted by Z-value)', pad=20)
plt.xlabel('INT Value', fontsize=12)
plt.ylabel('Network', fontsize=12)

# 调整布局
plt.tight_layout()
plt.show()
