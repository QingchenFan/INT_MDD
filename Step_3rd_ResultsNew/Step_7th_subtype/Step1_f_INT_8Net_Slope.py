import pandas as pd
import statsmodels.formula.api as smf
from nilearn.conftest import matplotlib

matplotlib.use('Agg')

# 1. 读取数据
df = pd.read_csv("/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step2_Ttest/HC_INT20_7net_agesex_FD.csv")

# 2. 识别网络列并转换为长格式 (Long format)
networks = ['subcortical', 'Visual', 'Somatomotor', 'Dorsal_Attention',
            'Ventral_Attention', 'Limbic', 'Frontoparietal', 'Default']

df_long = pd.melt(df,
                  id_vars=['subID', 'age', 'sex', 'mean_fd'],
                  value_vars=networks,
                  var_name='network',
                  value_name='INT_value')

# 3. 赋予 X 轴层级 (Rank)
rank_mapping = {
    'subcortical': 0,
    'Visual': 1,
    'Somatomotor': 2,
    'Dorsal_Attention': 3,
    'Ventral_Attention': 4,
    'Limbic': 5,
    'Frontoparietal': 6,
    'Default': 7
}
df_long['network_rank'] = df_long['network'].map(rank_mapping)

# 4. 拟合线性混合效应模型 (LMM)
# 固定效应: 评估网络层级的影响，并控制 age, sex, mean_fd
# 随机效应 (re_formula): 允许每个被试拥有自己独特的层级斜率 (network_rank)
model = smf.mixedlm("INT_value ~ network_rank + age + sex + mean_fd",
                    df_long,
                    groups=df_long["subID"],
                    re_formula="~network_rank")
result = model.fit()

# 5. 提取经 LMM "局部池化" 降噪后的个体稳健斜率 (Robust Slopes)
# 个体斜率 = 全局平均斜率 + 个体随机偏移量
fixed_slope = result.params['network_rank']
random_effects = pd.DataFrame(result.random_effects).T
robust_slopes = fixed_slope + random_effects['network_rank']

# 整理成初步的 DataFrame
slopes_df = robust_slopes.reset_index()
slopes_df.columns = ['subID', 'robust_INT_slope']

# ==========================================
# 6. 合并协变量信息 (新增修改部分)
# 从原始宽表 df 中提取 ID 及协变量，由于每个 subID 只有一行，直接取即可
covariates_df = df[['subID', 'age', 'sex', 'mean_fd']].drop_duplicates()

# 根据 'subID' 将斜率表和协变量表横向拼接
final_df = pd.merge(slopes_df, covariates_df, on='subID', how='left')
# ==========================================

# 7. 整理成最终 DataFrame 并保存
# 此时 final_df 包含了 ['subID', 'robust_INT_slope', 'age', 'sex', 'mean_fd']
final_df.to_csv("/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_INTSlope/HC_INT_slopes.csv", index=False)

print("数据保存成功！前5行预览：")
print(final_df.head())