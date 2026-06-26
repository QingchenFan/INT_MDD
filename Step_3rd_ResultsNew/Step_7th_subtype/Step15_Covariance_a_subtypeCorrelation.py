import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nilearn.conftest import matplotlib

from scipy.stats import pearsonr
from skbio.stats.distance import mantel
matplotlib.use('Agg')

# =====================================================
# 1. Load network
# =====================================================

net1 = pd.read_csv(
    "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_Covariation/subtype1_INTcovariance_network.csv",
    index_col=0
)

net2 = pd.read_csv(
    "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_Covariation/subtype2_INTcovariance_network.csv",
    index_col=0
)

# =====================================================
# 2. Extract upper triangle
# =====================================================

triu = np.triu_indices_from(net1, k=1)

edge1 = net1.values[triu]
edge2 = net2.values[triu]

print("Number of edges:", len(edge1))

# =====================================================
# 3. Edge Correlation
# =====================================================

r, _ = pearsonr(edge1, edge2)

print("\n==============================")
print("Edge Correlation")
print("==============================")
print(f"r = {r:.4f}")

# =====================================================
# 4. Frobenius Distance
# =====================================================

fro_dist = np.linalg.norm(
    net1.values - net2.values,
    ord='fro'
)

print("\n==============================")
print("Frobenius Distance")
print("==============================")
print(f"Distance = {fro_dist:.4f}")

# =====================================================
# 5. Mantel Test
# =====================================================

dist1 = 1 - net1.values
dist2 = 1 - net2.values

mantel_r, mantel_p, _ = mantel(
    dist1,
    dist2,
    method='pearson',
    permutations=5000
)

print("\n==============================")
print("Mantel Test")
print("==============================")
print(f"Mantel r = {mantel_r:.4f}")
print(f"Permutation p = {mantel_p:.6f}")

# =====================================================
# 6. Hexbin Plot
# =====================================================

plt.figure(figsize=(8, 8))

hb = plt.hexbin(
    edge1,
    edge2,
    gridsize=80,
    mincnt=1
)

plt.colorbar(
    hb,
    label="Number of edges"
)

# identity line

xmin = min(edge1.min(), edge2.min())
xmax = max(edge1.max(), edge2.max())

plt.plot(
    [xmin, xmax],
    [xmin, xmax],
    'k--',
    linewidth=2
)

plt.xlabel(
    "Subtype1 INT covariance",
    fontsize=14
)

plt.ylabel(
    "Subtype2 INT covariance",
    fontsize=14
)

plt.title(
    f"INT Covariance Network Similarity\n"
    f"Edge correlation = {r:.3f}",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_Covariation/result1_subtypeCorrelation/INT_network_similarity_hexbin.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# 7. Optional: Simple Scatter Plot
# =====================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    edge1,
    edge2,
    s=4,
    alpha=0.15
)

plt.plot(
    [xmin, xmax],
    [xmin, xmax],
    'r--',
    linewidth=2
)

plt.xlabel(
    "Subtype1 INT covariance",
    fontsize=14
)

plt.ylabel(
    "Subtype2 INT covariance",
    fontsize=14
)

plt.title(
    f"Edge Correlation r = {r:.3f}",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_Covariation/result1_subtypeCorrelation/INT_network_similarity_scatter.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()