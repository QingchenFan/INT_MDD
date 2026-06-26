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
# ⚠️ 注意：请确保您读取的这两个 csv 文件中确实包含 'mean_fd' 列
hc_path = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step3_aINT7Net_Diff/subtype1_INT_7net_agesex_FD.csv"
mdd_path = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step3_aINT7Net_Diff/subtype2_INT_7net_agesex_FD.csv"

hc = pd.read_csv(hc_path)
mdd = pd.read_csv(mdd_path)

hc["Group"] = "subtype1"
mdd["Group"] = "subtype2"

df = pd.concat([hc, mdd], ignore_index=True)

# =========================
# 2) Networks & Data Cleaning
# =========================
networks = ['subcortical', 'Visual', 'Somatomotor', 'Dorsal_Attention',
            'Ventral_Attention', 'Limbic', 'Frontoparietal', 'Default']

# 【修改点1】：剔除在感兴趣网络及协变量（包含 mean_fd）上存在缺失值的样本
df = df.dropna(subset=["age", "sex", "mean_fd"] + networks)
print("Data shape after dropna:", df.shape)

# =========================
# 3) GLM analysis: Feature ~ Group + age + C(sex) + mean_fd
# =========================
results = []

for net in networks:
    # 【修改点2】：将 mean_fd 添加到线性回归模型公式中
    formula = f"{net} ~ C(Group) + age + C(sex) + mean_fd"
    model = smf.ols(formula, data=df).fit()

    # 提取的是 subtype2 相对于 subtype1 的差异
    target_param = "C(Group)[T.subtype2]"

    beta = model.params.get(target_param, np.nan)
    tval = model.tvalues.get(target_param, np.nan)
    pval = model.pvalues.get(target_param, np.nan)

    results.append([net, beta, tval, pval])

res = pd.DataFrame(results, columns=["Network", "Beta(s2-s1)", "t", "p_uncorrected"])

# 进行 FDR 多重比较校正
res["p_FDR"] = multipletests(res["p_uncorrected"], method="fdr_bh")[1]

print("\n=== GLM Results (FDR Corrected, controlling for age, sex, mean_fd) ===")
print(res)

# Save GLM table
# 【修改点3】：保存表格文件名加上了 FD 标识
out_table = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step3_aINT7Net_Diff/S1_S2_INT8net_GLMagesexFD.csv"
res.to_csv(out_table, index=False)
print(f"\nSaved GLM results table to: {out_table}")

# =========================
# 4) Convert to long format for plotting
# =========================
plot_df = df.melt(
    id_vars=["Group"],
    value_vars=networks,
    var_name="Network",
    value_name="INT"
)

# 保证网络顺序不乱
plot_df["Network"] = pd.Categorical(plot_df["Network"], categories=networks, ordered=True)

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

star_dict = dict(zip(res["Network"], res["p_FDR"].apply(p_to_star)))

# =========================
# 6) Plot (single axis: shared x/y)
# =========================
sns.set(style="whitegrid")
plt.figure(figsize=(18, 7))

# 画箱线图
ax = sns.boxplot(
    data=plot_df,
    x="Network",
    y="INT",
    hue="Group",
    showfliers=False,
    palette="Set2"
)

# 画散点图
sns.stripplot(
    data=plot_df,
    x="Network",
    y="INT",
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
# 取前两个句柄（即 boxplot 生成的）
ax.legend(handles[:2], labels[:2], title="Group", loc="upper right", bbox_to_anchor=(1.1, 1))

# 获取数据的全局 Y 轴最大/最小值，用于计算连线高度
y_max = plot_df["INT"].max()
y_range = y_max - plot_df["INT"].min()
h = y_range * 0.015  # 连线向下的小尾巴长度

# 添加显著性星号和对比连线
for i, net in enumerate(networks):
    star = star_dict.get(net, "")

    # 只有当两组间存在显著差异 (不为 ns 或空) 时才画线标星
    if star and star != "ns":
        # 获取该网络中实际数据的最高点，作为基准高度
        net_max = plot_df[plot_df["Network"] == net]["INT"].max()
        line_y = net_max + y_range * 0.03

        # seaborn hue 偏移量：左侧箱体大概在 i - 0.2, 右侧箱体在 i + 0.2
        x1, x2 = i - 0.2, i + 0.2

        # 画横线和小尾巴
        plt.plot([x1, x1, x2, x2], [line_y, line_y + h, line_y + h, line_y], lw=1.5, color='black')
        # 写上星号
        ax.text((x1 + x2) / 2, line_y + h, star, ha="center", va="bottom", color="black", fontsize=15,
                fontweight='bold')

# 调整 Y 轴上限，防止星号被图框切掉
plt.ylim(plot_df["INT"].min() - y_range * 0.05, y_max + y_range * 0.15)

# 标题和轴标签
# 【修改点4】：标题中加入了 mean_fd 说明
ax.set_title("INT Comparison between Subtype1 and Subtype2 across 8 Networks\n(Covariates: Age, Sex, mean_fd)", fontsize=16,
             pad=20)
ax.set_xlabel("Yeo 7 Networks + Subcortical", fontsize=13)
ax.set_ylabel("INT Value", fontsize=13)

plt.xticks(rotation=15, ha="center")
plt.tight_layout()

# Save figure
# 【修改点3】：保存图像文件名加上了 FD 标识
out_fig = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step3_aINT7Net_Diff/S1_S2_INT8net_GLMagesexFD.png"
plt.savefig(out_fig, dpi=300, bbox_inches="tight")
print(f"\nFigure saved to: {out_fig}")