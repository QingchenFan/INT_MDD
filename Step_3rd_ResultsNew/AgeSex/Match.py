import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

# =========================
# 1) Load matched dataset
# =========================
data_path = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/AgeSex/matched_1to1_dataset.csv"
df = pd.read_csv(data_path)


# =========================
# 2) Define groups
# =========================
hc = df[df["Group"] == 0].copy()
mdd = df[df["Group"] == 1].copy()

print("\nSample size:")
print("HC :", hc.shape[0])
print("MDD:", mdd.shape[0])

# =========================
# 3) Age difference (Welch t-test)
# =========================
t_age, p_age = ttest_ind(mdd["age"], hc["age"], equal_var=False)

print("\nAge comparison (Welch t-test):")
print("HC  mean ± SD:", np.mean(hc["age"]), "±", np.std(hc["age"], ddof=1))
print("MDD mean ± SD:", np.mean(mdd["age"]), "±", np.std(mdd["age"], ddof=1))
print("t =", t_age, ", p =", p_age)

# =========================
# 4) Sex distribution difference
# Chi-square test (or Fisher if small counts)
# =========================
cont_table = pd.crosstab(df["Group"], df["sex"])

print("\nSex contingency table (Group x sex):")
print(cont_table)

# check if any expected count < 5
chi2, p_chi, dof, expected = chi2_contingency(cont_table)

if (expected < 5).any():
    print("\nExpected count < 5 detected -> using Fisher exact test.")
    # Fisher exact only works for 2x2 tables
    if cont_table.shape == (2, 2):
        oddsratio, p_sex = fisher_exact(cont_table.values)
        print("Fisher exact test p =", p_sex)
    else:
        p_sex = p_chi
        print("Warning: table not 2x2, fallback to Chi-square p =", p_sex)
else:
    p_sex = p_chi
    print("\nChi-square test:")
    print("chi2 =", chi2, ", dof =", dof, ", p =", p_sex)

# =========================
# 5) Network comparison (Welch t-test)
# =========================
networks = [
    "subcortical", "Visual", "Somatomotor", "Dorsal_Attention",
    "Ventral_Attention", "Limbic", "Frontoparietal", "Default"
]

results = []

for net in networks:
    hc_values = hc[net].values
    mdd_values = mdd[net].values

    tval, pval = ttest_ind(mdd_values, hc_values, equal_var=False)

    results.append([
        net,
        np.mean(hc_values),
        np.mean(mdd_values),
        tval,
        pval
    ])

res = pd.DataFrame(results, columns=["Network", "Mean_HC", "Mean_MDD", "t", "p"])

# =========================
# 6) FDR correction (BH)
# =========================
res["p_FDR"] = multipletests(res["p"], method="fdr_bh")[1]

print("\nNetwork comparison results (MDD vs HC):")
print(res)

# =========================
# 7) Save results
# =========================
out_path = "./network_comparison_results_FDR.csv"
#res.to_csv(out_path, index=False)

print("\nSaved network comparison results to:", out_path)