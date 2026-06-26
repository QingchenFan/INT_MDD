import pandas as pd
from scipy.stats import chi2_contingency

# 1. 读取两个亚型的CSV数据
df1 = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step8_FirstEp_Diff/subtype1_FirstEp.csv')
df2 = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step8_FirstEp_Diff/subtype2_FirstEp.csv')

# 2. 统计两个亚型中首发(1)和复发(2)的人数
# sort_index() 确保统计的顺序是 1 在前，2 在后
counts1 = df1['FirstEpisode'].value_counts().sort_index()
counts2 = df2['FirstEpisode'].value_counts().sort_index()

print("=== 亚型 1 统计 ===")
print(counts1)
print("\n=== 亚型 2 统计 ===")
print(counts2)

# 3. 构建列联表 (Contingency Table)
table = [counts1.tolist(), counts2.tolist()]

# 4. 进行卡方检验
chi2, p, dof, expected = chi2_contingency(table)

print(f"\n=== 统计检验结果 ===")
print(f"卡方值 (Chi-square): {chi2:.4f}")
print(f"自由度 (Degrees of freedom): {dof}")
print(f"P值 (p-value): {p:.4f}")

# 判断是否显著
if p < 0.05:
    print("结论: P < 0.05, 两个亚型的首发/复发分布存在显著差异。")
else:
    print("结论: P >= 0.05, 两个亚型的首发/复发分布不存在显著差异。")