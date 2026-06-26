import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import matplotlib

# 设置绘图风格
matplotlib.use('Agg')
sns.set(style="whitegrid")

# ==========================================
# 1. 加载数据
# ==========================================
hc = pd.read_csv("/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step3_Structure/HC_GMV_7Net_agesex.csv")
mdd = pd.read_csv("/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype2_GMV_7Net.csv")

hc["Group"] = "HC"
mdd["Group"] = "MDD"

# ==========================================
# 2. 1:1 最近邻匹配 (年龄+性别)
# ==========================================
def perform_strict_matching(df_hc, df_mdd):
    matched_hc_idx = []
    matched_mdd_idx = []

    # 按性别分层匹配，确保性别比例完全一致
    for sex_val in df_hc['sex'].unique():
        sub_hc = df_hc[df_hc['sex'] == sex_val].copy()
        sub_mdd = df_mdd[df_mdd['sex'] == sex_val].copy()

        if sub_hc.empty or sub_mdd.empty:
            continue

        # 计算两组间的年龄距离矩阵
        dist_matrix = np.abs(sub_mdd['age'].values[:, None] - sub_hc['age'].values[None, :])

        # 贪婪匹配算法 (寻找全局或局部最优配对)
        mdd_indices = sub_mdd.index.tolist()
        hc_indices = sub_hc.index.tolist()

        pairs = []
        for i in range(len(mdd_indices)):
            for j in range(len(hc_indices)):
                pairs.append((dist_matrix[i, j], mdd_indices[i], hc_indices[j]))

        # 按距离从小到大排序
        pairs.sort(key=lambda x: x[0])

        seen_mdd = set()
        seen_hc = set()
        for dist, m_idx, h_idx in pairs:
            if m_idx not in seen_mdd and h_idx not in seen_hc:
                matched_mdd_idx.append(m_idx)
                matched_hc_idx.append(h_idx)
                seen_mdd.add(m_idx)
                seen_hc.add(h_idx)

    return df_hc.loc[matched_hc_idx], df_mdd.loc[matched_mdd_idx]

# 执行匹配
hc_matched, mdd_matched = perform_strict_matching(hc, mdd)
df_matched = pd.concat([hc_matched, mdd_matched], ignore_index=True)

# ==========================================
# 3. 协变量差异分析 (匹配前 vs 匹配后)
# ==========================================
# 检验年龄
t_age_pre, p_age_pre = ttest_ind(hc['age'], mdd['age'])
t_age_post, p_age_post = ttest_ind(hc_matched['age'], mdd_matched['age'])

# 检验 TIV
t_tiv_pre, p_tiv_pre = ttest_ind(hc['TIV'], mdd['TIV'])
t_tiv_post, p_tiv_post = ttest_ind(hc_matched['TIV'], mdd_matched['TIV'])

print(f"匹配前样本量: HC={len(hc)}, MDD={len(mdd)}")
print(f"匹配后样本量: 每组 {len(hc_matched)} 人")
print("-" * 40)
print(f"匹配前年龄差异: t={t_age_pre:.4f}, p={p_age_pre:.4f}")
print(f"匹配后年龄差异: t={t_age_post:.4f}, p={p_age_post:.4f}")
print("-" * 40)
print(f"匹配前 TIV 差异: t={t_tiv_pre:.4f}, p={p_tiv_pre:.4f}")
print(f"匹配后 TIV 差异: t={t_tiv_post:.4f}, p={p_tiv_post:.4f}")
print("-" * 40)

# ==========================================
# 4. 脑网络差异分析 (基于匹配后的数据)
# ==========================================
networks = [
    'subcortical_GMV', 'Visual_GMV', 'Somatomotor_GMV', 'Dorsal_Attention_GMV',
    'Ventral_Attention_GMV', 'Limbic_GMV', 'Frontoparietal_GMV', 'Default_GMV'
]
results = []
for net in networks:
    # 【修改点 1】：建立 GLM 模型，控制残余的 年龄、性别 以及 TIV 影响
    formula = f"{net} ~ C(Group, Treatment('HC')) + age + sex + TIV"
    model = smf.ols(formula, data=df_matched).fit()

    beta = model.params.get("C(Group, Treatment('HC'))[T.MDD]", np.nan)
    t_val = model.tvalues.get("C(Group, Treatment('HC'))[T.MDD]", np.nan)
    p_val = model.pvalues.get("C(Group, Treatment('HC'))[T.MDD]", np.nan)
    results.append([net, beta, t_val, p_val])

res_df = pd.DataFrame(results, columns=["Network", "Beta(MDD-HC)", "t", "p"])
# FDR 校正
res_df["p_FDR"] = multipletests(res_df["p"], method="fdr_bh")[1]

print("\n网络差异分析结果 (控制 age, sex, TIV 匹配后):")
print(res_df)

# ==========================================
# 5. 可视化
# ==========================================
# 【修改点 2】：将 value_name 改为 GMV
plot_df = df_matched.melt(id_vars=["Group"], value_vars=networks,
                          var_name="Network", value_name="GMV")
plot_df["Network"] = pd.Categorical(plot_df["Network"], categories=networks, ordered=True)

plt.figure(figsize=(16, 8))
# 箱线图
ax = sns.boxplot(data=plot_df, x="Network", y="GMV", hue="Group",
                 showfliers=False, palette="Set2")
# 散点图
sns.stripplot(data=plot_df, x="Network", y="GMV", hue="Group",
              dodge=True, jitter=True, alpha=0.3, size=3, palette="Set2", legend=False)

# 添加显著性星号 (基于 p_FDR)
def p_to_star(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""

# 【修改点 3】：基于 GMV 动态调整星号的 Y 轴高度位置
y_max_vals = plot_df.groupby("Network")["GMV"].max()
y_range = plot_df["GMV"].max() - plot_df["GMV"].min()
offset = y_range * 0.03  # 设置 3% 的悬浮间隙

for i, net in enumerate(networks):
    p_fdr = res_df.loc[res_df["Network"] == net, "p_FDR"].values[0]
    star = p_to_star(p_fdr)
    if star:
        y_pos = y_max_vals[net] + offset
        ax.text(i, y_pos, star, ha='center', va='bottom', color='red', fontsize=18, fontweight='bold')

plt.title(f"GMV Differences After Age/Sex Matching (Controlled for TIV)\nN={len(hc_matched)} per group", fontsize=16)
plt.xlabel("Yeo 7 Networks + Subcortical", fontsize=14)
plt.ylabel("Gray Matter Volume (GMV)", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# 保存结果
plt.savefig("/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/s2_matched_network_diff_TIV.png", dpi=300)
res_df.to_csv("/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/s2_matched_glm_results_TIV.csv", index=False)

print("\n计算和绘图完成！已保存结果至相应目录。")
