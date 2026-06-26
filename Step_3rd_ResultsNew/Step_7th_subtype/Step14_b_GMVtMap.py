import pandas as pd
import statsmodels.api as sm

# 1. 读取两组数据
# 假设已经将文件放在同一目录下
df_sub = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype2_GMV246.csv')
df_hc = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/HC_GrayVol246.csv')

# 2. 添加分组标签
# (通常将患者组设为 1，健康对照组设为 0)
# 这样计算出来的正 t 值代表患者灰质体积增加，负 t 值代表灰质体积减少
df_sub['Group'] = 1
df_hc['Group'] = 0

# 3. 合并数据表
df = pd.concat([df_sub, df_hc], ignore_index=True)

# 4. 定义需要排除的非脑区特征列，剩下的列即为脑区(246个)
exclude_cols = ['subID', 'TIV', 'age', 'sex', 'Group']
regions = [col for col in df.columns if col not in exclude_cols]

# 5. 循环遍历每一个脑区，进行多元线性回归
t_values = {}
p_values = {}

# 构建自变量 X：包括我们要考察的组别(Group)，以及需要控制的协变量(age, sex, TIV)
X = df[['Group', 'age', 'sex', 'TIV']]
X = sm.add_constant(X)  # 必须添加常数项（截距）

for region in regions:
    # 因变量 y 为当前循环到的脑区灰质体积
    y = df[region]

    # 拟合 OLS (普通最小二乘法) 模型
    model = sm.OLS(y, X, missing='drop').fit()

    # 从模型结果中专门提取 'Group' 这一项主效应的 t 值和 p 值
    t_values[region] = model.tvalues['Group']
    p_values[region] = model.pvalues['Group']

# 6. 将提取出来的全脑 t 值合并为一个数据表 (这就是您的 unthresholded t-map)
t_map_df = pd.DataFrame({
    'Region': list(t_values.keys()),
    't_value': list(t_values.values()),
    'p_value': list(p_values.values())
})

# 7. 保存结果
t_map_df.to_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step16_Epicenters/subtype2_t_map_results.csv', index=False)
print("T-map 计算完成！")