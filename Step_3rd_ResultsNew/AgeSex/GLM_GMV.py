import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use('Agg')
# =========================
# 1) Read data
# =========================
s1_path = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype1_GMV_7Net.csv"
s2_path = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype2_GMV_7Net.csv"

# 修正变量名 hc -> s1, mdd -> s2
s1 = pd.read_csv(s1_path)
s2 = pd.read_csv(s2_path)

# 修正组别标签
s1["Group"] = "subtype1"
s2["Group"] = "subtype2"

df = pd.concat([s1, s2], ignore_index=True)

print("Data shape:", df.shape)

# =========================
# 2) Networks
# =========================
# GMV 网络名称
networks = [
    'subcortical_GMV', 'Visual_GMV', 'Somatomotor_GMV', 'Dorsal_Attention_GMV',
    'Ventral_Attention_GMV', 'Limbic_GMV', 'Frontoparietal_GMV', 'Default_GMV'
]

# =========================
# 3) GLM analysis: Feature ~ Group + age + C(sex) + TIV
# =========================
results = []

for net in networks:
    # 强制将 sex 作为分类变量，并加入 TIV 控制头颅大小差异
    formula = f"{net} ~ C(Group) + age + C(sex) + TIV"
    model = smf.ols(formula, data=df).fit()

    # 修正提取参数的键值：基于模型设定，此时提取的是 subtype2 相比于 subtype1 的差异
    target_param = "C(Group)[T.subtype2]"

    beta = model.params.get(target_param, np.nan)
    tval = model.tvalues.get(target_param, np.nan)
    pval = model.pvalues.get(target_param, np.nan)

    results.append([net, beta, tval, pval])

# 修正表格列名
res = pd.DataFrame(results, columns=["Network", "Beta(s2-s1)", "t", "p_uncorrected"])
res["p_FDR"] = multipletests(res["p_uncorrected"], method="fdr_bh")[1]

print("\n=== GLM Results (controlled for age, sex, TIV) ===")
print(res)

# Save GLM table
out_table = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step3_bGMV7Net_Diff/GMV_S1_S2_8net_GLMagesexTIV.csv"
res.to_csv(out_table, index=False)
print("\nSaved GLM results table to:", out_table)

# =========================
# 4) Convert to long format for plotting
# =========================
plot_df = df.melt(
    id_vars=["Group"],
    value_vars=networks,
    var_name="Network",
    value_name="GMV"
)

# 清理一下图表上的网络名称后缀，让 X 轴更好看 (可选)
plot_df['Network_Label'] = plot_df['Network'].str.replace('_GMV', '')
label_order = [net.replace('_GMV', '') for net in networks]

plot_df["Network_Label"] = pd.Categorical(plot_df["Network_Label"], categories=label_order, ordered=True)


# =========================
# 5) Prepare significance stars based on FDR p
# =========================
def p_to_star(p):
    if pd.isna(p): return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


star_dict = dict(zip(label_order, res["p_FDR"].apply(p_to_star)))

# =========================
# 6) Plot (single axis: shared x/y)
# =========================
sns.set(style="whitegrid")
plt.figure(figsize=(18, 7))

# 箱线图
ax = sns.boxplot(
    data=plot_df,
    x="Network_Label",
    y="GMV",
    hue="Group",
    showfliers=False,
    palette="Set2"
)

# 散点图
sns.stripplot(
    data=plot_df,
    x="Network_Label",
    y="GMV",
    hue="Group",
    dodge=True,
    jitter=True,
    alpha=0.5,
    size=4,
    linewidth=0,
    palette="Set2"
)

# 处理 legend 重复
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], title="Group", loc="upper right", bbox_to_anchor=(1.1, 1))

# --- 添加标准的显著性连线和星号 ---
y_max_global = plot_df["GMV"].max()
y_range = y_max_global - plot_df["GMV"].min()
h = y_range * 0.015  # 连线向下折角的小尾巴长度

for i, net_label in enumerate(label_order):
    star = star_dict.get(net_label, "")
    # 只有当经过 FDR 校正且差异显著时，才画线标星号
    if star and star != "ns":
        # 获取当前网络数据的最高点，在此基础上进行偏移
        net_max = plot_df[plot_df["Network_Label"] == net_label]["GMV"].max()
        line_y = net_max + y_range * 0.03

        # seaborn hue 在双组情况下的 x 轴相对偏移量大概是 ±0.2
        x1, x2 = i - 0.2, i + 0.2

        # 画倒 U 型连线
        plt.plot([x1, x1, x2, x2], [line_y, line_y + h, line_y + h, line_y], lw=1.5, color='black')
        # 在连线中央写上星号
        ax.text((x1 + x2) / 2, line_y + h, star, ha="center", va="bottom", color="black", fontsize=15,
                fontweight='bold')

# 动态调整 Y 轴上限，留出显示星号的空间
plt.ylim(plot_df["GMV"].min() - y_range * 0.05, y_max_global + y_range * 0.15)

# 标题和轴标签 (修正为了 Subtype1 和 Subtype2)
ax.set_title("GMV Comparison between Subtype1 and Subtype2 across 8 Networks\n(Covariates: Age, Sex, TIV)", fontsize=16,
             pad=20)
ax.set_xlabel("Yeo 7 Networks + Subcortical", fontsize=13)
ax.set_ylabel("Gray Matter Volume (GMV)", fontsize=13)

plt.xticks(rotation=15, ha="center")
plt.tight_layout()

# Save figure
out_fig = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step3_bGMV7Net_Diff/GMV_S1_S2_8net_GLMagesexTIV.png"
plt.savefig(out_fig, dpi=300, bbox_inches="tight")
print(f"\nFigure saved to: {out_fig}")