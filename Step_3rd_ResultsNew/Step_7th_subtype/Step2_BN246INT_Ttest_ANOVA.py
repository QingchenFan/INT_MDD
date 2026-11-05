from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests


# ==========================================================
# 1. 只需修改这里：三个输入 CSV 路径、一个输出文件夹路径
# ==========================================================
hc_path = "//HC_INT20_agesex_FD.csv"
subtype1_path = "//subtype1_INT20_agesex_FD.csv"
subtype2_path = "//subtype2_INT20_agesex_FD.csv"

output_dir = "//ANCOVA_results"


# ==========================================================
# 2. 读取数据、检查并合并
# ==========================================================
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

paths = {
    "HC": hc_path,
    "subtype1": subtype1_path,
    "subtype2": subtype2_path,
}

covariates = ["subID", "age", "sex", "mean_fd"]
frames = []
brain_regions = None

for group, path in paths.items():
    data = pd.read_csv(path, dtype={"subID": str})

    # 检查必需列
    missing_columns = set(covariates) - set(data.columns)
    if missing_columns:
        raise ValueError(f"{group} 缺少列：{missing_columns}")

    # 其余列均视为脑区列
    regions = [c for c in data.columns if c not in covariates]

    if brain_regions is None:
        brain_regions = regions
    elif set(regions) != set(brain_regions):
        raise ValueError(f"{group} 的脑区列与其他组不一致")

    # 检查受试者编号
    if data["subID"].isna().any() or data["subID"].duplicated().any():
        raise ValueError(f"{group} 存在缺失或重复的 subID")

    data["Group"] = group
    frames.append(data)

combined_df = pd.concat(frames, ignore_index=True)

if combined_df["subID"].duplicated().any():
    raise ValueError("三组之间存在重复 subID，请核实是否为同一受试者")

analysis_columns = brain_regions + ["age", "sex", "mean_fd"]

if combined_df[analysis_columns].isna().any().any():
    raise ValueError("分析数据中存在缺失值，请先处理后再运行")

numeric_columns = brain_regions + ["age", "mean_fd"]
combined_df[numeric_columns] = combined_df[numeric_columns].apply(
    pd.to_numeric, errors="raise"
)

if not np.isfinite(
    combined_df[numeric_columns].to_numpy(dtype=float)
).all():
    raise ValueError("脑区或协变量中存在无穷值")

combined_df["Group"] = pd.Categorical(
    combined_df["Group"],
    categories=["HC", "subtype1", "subtype2"]
)

print("各组样本量：")
print(combined_df.groupby("Group", observed=True).size())
print(f"\n脑区数量：{len(brain_regions)}")


# ==========================================================
# 3. 每个脑区拟合三组 ANCOVA，并计算三个事后对比
# ==========================================================
group_term = 'C(Group, Treatment(reference="HC"))'

coef_subtype1 = f"{group_term}[T.subtype1]"
coef_subtype2 = f"{group_term}[T.subtype2]"

omnibus_results = []
posthoc_results = []

for region in brain_regions:

    # repr() 配合 Q()，兼容数字开头、带特殊字符的脑区名称
    formula = (
        f"Q({region!r}) ~ {group_term} + age + C(sex) + mean_fd"
    )

    # 常规 OLS：与此前分析使用的模型一致
    model = smf.ols(formula, data=combined_df).fit()

    if np.linalg.matrix_rank(model.model.exog) < model.model.exog.shape[1]:
        raise ValueError(f"{region} 的设计矩阵不满秩，请检查协变量")

    # ---------- 总体组效应：Type II ANCOVA ----------
    anova_table = anova_lm(model, typ=2)
    group_result = anova_table.loc[group_term]

    omnibus_results.append({
        "ROI": region,
        "n_total": int(model.nobs),
        "F": float(group_result["F"]),
        "df_group": float(group_result["df"]),
        "df_resid": float(model.df_resid),
        "pvalue": float(group_result["PR(>F)"]),
    })

    # ---------- 构造三个线性对比 ----------
    # 以 HC 为参照：
    # beta1 = subtype1 - HC
    # beta2 = subtype2 - HC
    v1 = np.zeros(len(model.params))
    v2 = np.zeros(len(model.params))

    v1[model.params.index.get_loc(coef_subtype1)] = 1
    v2[model.params.index.get_loc(coef_subtype2)] = 1

    contrasts = {
        "HC_minus_subtype1": -v1,
        "HC_minus_subtype2": -v2,
        "subtype1_minus_subtype2": v1 - v2,
    }

    for contrast_name, contrast_vector in contrasts.items():

        # 同一个三组模型中的双侧 t 检验
        test = model.t_test(contrast_vector)
        ci_low, ci_high = test.conf_int(alpha=0.05).ravel()

        posthoc_results.append({
            "ROI": region,
            "contrast": contrast_name,
            "adjusted_mean_difference": np.asarray(test.effect).item(),
            "SE": np.asarray(test.sd).item(),
            "tvalue": np.asarray(test.tvalue).item(),
            "df_resid": float(model.df_resid),
            "pvalue": np.asarray(test.pvalue).item(),
            "CI95_low_unadjusted": ci_low,
            "CI95_high_unadjusted": ci_high,
            "n_total": int(model.nobs),
        })


# ==========================================================
# 4. 多重比较校正
# ==========================================================
omnibus_df = pd.DataFrame(omnibus_results)
posthoc_df = pd.DataFrame(posthoc_results)

# 总体检验：246 个脑区时，对全部 246 个 p 值统一 BH-FDR
omnibus_df["fdr_pvalue"] = multipletests(
    omnibus_df["pvalue"],
    alpha=0.05,
    method="fdr_bh"
)[1]

omnibus_df["significant_fdr_005"] = omnibus_df["fdr_pvalue"] < 0.05

# 事后检验：246 个脑区时，对全部 738 个 p 值统一 BH-FDR
# 不先筛选总体显著脑区，也不按三个对比分别校正
posthoc_df["fdr_pvalue"] = multipletests(
    posthoc_df["pvalue"],
    alpha=0.05,
    method="fdr_bh"
)[1]

posthoc_df["significant_fdr_005"] = posthoc_df["fdr_pvalue"] < 0.05

# 附上对应脑区的总体检验结果，便于查看
posthoc_df = posthoc_df.merge(
    omnibus_df[
        ["ROI", "fdr_pvalue", "significant_fdr_005"]
    ].rename(columns={
        "fdr_pvalue": "omnibus_fdr_pvalue",
        "significant_fdr_005": "omnibus_significant_fdr_005",
    }),
    on="ROI",
    how="left",
    validate="many_to_one",
)


# ==========================================================
# 5. 保存 CSV
# ==========================================================
omnibus_df.to_csv(
    output_dir / "ANCOVA_omnibus.csv",
    index=False,
    encoding="utf-8-sig"
)

posthoc_df.to_csv(
    output_dir / "posthoc_all.csv",
    index=False,
    encoding="utf-8-sig"
)

# 分开保存三个对比，保留全部事后检验统一校正后的 p 值
for contrast_name, result in posthoc_df.groupby("contrast", sort=False):
    result.to_csv(
        output_dir / f"posthoc_{contrast_name}.csv",
        index=False,
        encoding="utf-8-sig"
    )


# ==========================================================
# 6. 打印结果摘要
# ==========================================================
print(
    f"\n总体组效应 FDR < 0.05："
    f"{omnibus_df['significant_fdr_005'].sum()} / "
    f"{len(omnibus_df)} 个脑区"
)

print(
    f"\n事后检验统一校正数量：{len(posthoc_df)}"
)

for contrast_name, result in posthoc_df.groupby("contrast", sort=False):
    print(
        f"{contrast_name}："
        f"{result['significant_fdr_005'].sum()} / "
        f"{len(result)} 个脑区 FDR < 0.05"
    )

print(f"\n结果已保存至：{output_dir.resolve()}")