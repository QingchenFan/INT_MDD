import pandas as pd
from scipy import stats

# 1. 加载数据
hc_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Test/HC_INT20_7net_agesex.csv')
mdd_df = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Test/MDD_INT20_7net_agesex_I.csv')

# --- 2. 年龄 (Age) 的 t 检验 ---
# 提取年龄数据并排除缺失值
hc_age = hc_df['age'].dropna()
mdd_age = mdd_df['age'].dropna()

# 执行独立样本 t 检验 (使用 Welch's t-test，不假设方差齐性)
t_stat, p_val_age = stats.ttest_ind(hc_age, mdd_age, equal_var=False)

# --- 3. 性别 (Sex) 的卡方检验 ---
# 统计两组中各性别的频数
hc_sex_counts = hc_df['sex'].value_counts().sort_index()
mdd_sex_counts = mdd_df['sex'].value_counts().sort_index()

# 构建列联表 (Contingency Table)
contingency_table = pd.DataFrame([hc_sex_counts, mdd_sex_counts], index=['HC', 'MDD']).fillna(0)

# 执行卡方检验
chi2, p_val_sex, dof, expected = stats.chi2_contingency(contingency_table)

# --- 4. 打印结果 ---
print("=== 年龄 (Age) t 检验结果 ===")
print(f"健康对照组 (HC): {hc_age.mean():.2f} ± {hc_age.std():.2f}")
print(f"重度抑郁组 (MDD): {mdd_age.mean():.2f} ± {mdd_age.std():.2f}")
print(f"t 统计量: {t_stat:.4f}, p 值: {p_val_age:.4f}")

print("\n=== 性别 (Sex) 卡方检验结果 ===")
print("性别频数统计表 (1=男, 2=女/或其他码值):")
print(contingency_table)
print(f"卡方统计量: {chi2:.4f}, p 值: {p_val_sex:.4f}")