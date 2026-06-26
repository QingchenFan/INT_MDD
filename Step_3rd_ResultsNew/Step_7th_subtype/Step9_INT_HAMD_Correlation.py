import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# 1. 读取数据
# 假设文件名为 'subtype1_mean_INT_HAMD.csv' 且与代码在同一目录下
df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step9_INT_HAMD_Correlation/'
                 'subtype2_mean_INT_HAMD.csv')

# 2. 计算皮尔逊相关系数和 P 值
corr, p_val = pearsonr(df['mean_INT'], df['HAMD_0w'])
print(f"Correlation (r): {corr:.4f}")
print(f"P-value (p): {p_val:.4e}")

# 3. 绘制散点图并添加回归线
plt.figure(figsize=(8, 6))
sns.regplot(
    data=df,
    x='mean_INT',
    y='HAMD_0w',
    scatter_kws={'alpha': 0.7},  # 设置散点透明度
    line_kws={'color': 'red'}    # 设置拟合线颜色为红色
)

# 4. 设置图表标题和坐标轴标签
plt.title(f'Scatter Plot of mean_INT vs HAMD_0w\n(r = {corr:.3f}, p = {p_val:.3e})')
plt.xlabel('mean_INT')
plt.ylabel('HAMD_0w')

# 5. 添加网格线并优化布局
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 6. 保存图片 (如果需要直接在本地查看，可以将 savefig 替换为 plt.show())
plt.savefig('scatter_plot.png')