import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# 从 CSV 文件中读取 MDD 数据
df_mdd = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_BP135MDD/volume/INTvalue_MDDgroup_net7.csv')
# 添加组别标识
df_mdd['Group'] = 'MDD'

# 从 CSV 文件中读取 HC 数据
df_hc = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_BP135MDD/volume/INTvalue_HCgroup_net7.csv')
# 添加组别标识
df_hc['Group'] = 'HC'

# 合并数据
df_combined = pd.concat([df_mdd, df_hc], ignore_index=True)

# 将数据转换为长格式
df_long = df_combined.melt(id_vars=['subID', '0', 'Group'],
                           value_vars=['Visual', 'Somatomotor', 'Dorsal Attention',
                                       'Ventral Attention', 'Limbic', 'Frontoparietal', 'Default'],
                           var_name='Network',
                           value_name='Value')

# 指定网络顺序（从上到下）
network_order = ['Visual', 'Somatomotor', 'Dorsal Attention',
                 'Ventral Attention', 'Limbic', 'Frontoparietal', 'Default']

plt.figure(figsize=(10, 8))

# 绘制横向箱体图，通过 hue 参数区分不同组
sns.boxplot(x='Value',
            y='Network',
            hue='Group',
            data=df_long,
            order=network_order,
            orient='h',
            palette='vlag',
            width=0.7)

# 添加标题和标签
plt.title('Distribution of Network Values across Subjects by Group', pad=20)
plt.xlabel('INT Value')
plt.ylabel('Network')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout()
# 显示图例
plt.legend(title='Group', loc='lower right')

plt.show()