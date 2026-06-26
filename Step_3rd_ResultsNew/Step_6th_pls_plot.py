import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn.conftest import matplotlib

matplotlib.use('Agg')
# 读取CSV文件
data_subtype1 = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_scale/PLSR_Output/run_66_r021_p0032/prediction.csv')


# 创建一个图和两个轴对象
fig, ax = plt.subplots(figsize=(10, 6))
ax.grid(False)

# ========== 新增：去掉背景的核心代码 ==========
# 设置整个画布背景为白色（也可设为None实现透明）
fig.patch.set_facecolor('white')
# 设置坐标轴区域背景为白色（去掉默认的浅灰色背景）
ax.set_facecolor('white')

# 在第一个轴上绘制第一个数据集的散点回归图，增加scatter_kws中的alpha参数设置透明度，设置ci=95显示95%置信区间
sns.regplot(x='y_true', y='y_pred', data=data_subtype1, ax=ax,
            scatter_kws={'color': '#003366', 'alpha': 0.8,'s':150},  # 设置散点透明度为0.6
            line_kws={'color': '#003366','linewidth':2.5}, ci=95)  # 添加95%置信区间

# 设置图表标题和坐标轴标签
ax.set_xlabel('Actual score', size=36)
ax.set_ylabel('Predicted score', size=36)
#Prediction score
# 自定义x轴和y轴的刻度
ax.set_xticks([0.0,  0.5,   1.0])
ax.set_yticks([0.2, 0.4,  0.6, 0.8])

# 增大刻度标签的字体大小
plt.xticks(fontsize=36)  # 增大x轴刻度字体大小
plt.yticks(fontsize=36)  # 增大y轴刻度字体大小

# 设置坐标轴线条属性
for axis in ['left', 'bottom']:
    ax.spines[axis].set_color('black')
    ax.spines[axis].set_linewidth(3)

# 去掉x轴和y轴上的小刻度线
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

# 去掉上边线和右边线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 保存和显示
plt.tight_layout()  # 确保所有元素都适合图形区域
plt.savefig('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_scale/PLS_r021_p00031.png', dpi=300)
plt.show()