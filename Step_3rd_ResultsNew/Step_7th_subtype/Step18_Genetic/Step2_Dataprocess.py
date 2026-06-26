import pandas as pd
import numpy as np
import abagen
from scipy.stats import spearmanr
import os
from statsmodels.stats.multitest import fdrcorrection
from neuromaps import nulls, stats, parcellate
from neuromaps.images import dlabel_to_gifti  # 新增：转换 dlabel
import nibabel as nib
from nilearn.image import new_img_like
import pandas as pd
import numpy as np
import abagen
from abagen import keep_stable_genes
import os
# ==========================================
# 路径设置（新增表面 atlas 路径）
# ==========================================
gmv_data_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step18_Genetic/result1_zmap/subtype1_z_map.csv'
atlas_path = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_246_2mm.nii.gz'
lut_path = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_freesurfer/BN_Atlas_246_LUT.txt'
surface_atlas_path = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii'  # ← 替换为你的实际路径

output_path = './gene_imaging_correlation_results.csv'
stable_matrix_path = './stable_gene_expression.csv'
print("--- 正在启动分析流程 ---")

print("--- 启动 Imaging-Transcriptomics 分析流程 ---")

# ==========================================
# 2. 读取无表头的 Z-map 数据
# ==========================================

df_gmv = pd.read_csv(gmv_data_path, header=None)
gmv_vector = df_gmv.iloc[:, 1].values


# ==========================================
# 3. 构建图谱信息 (Atlas Info)
# ==========================================
print("构建 Brainnetome 图谱信息...")
df_lut = pd.read_csv(
    lut_path,
    sep=r"\s+",
    header=None,
    skiprows=1,
    names=["ID", "Label", "R", "G", "B", "A"]
)

df_lut["ID"] = df_lut["ID"].astype(int)
df_lut["hemisphere"] = df_lut["Label"].apply(lambda x: "left" if "_L" in x else "right")
# 根据 Brainnetome 约定，1-210 为皮层，211-246 为皮层下
df_lut["structure"] = df_lut["ID"].apply(lambda x: "cortex" if x <= 210 else "subcortex")

atlas_info = df_lut[["ID", "hemisphere", "structure"]].copy()
atlas_info.columns = ["id", "hemisphere", "structure"]
atlas_info.set_index("id", inplace=True)

# ==========================================
# 4. Abagen: 提取艾伦脑图谱表达数据
# ==========================================
print("开始连接 AHBA 数据库提取基因表达数据 (可能需要数分钟)...")
expression_return = abagen.get_expression_data(
    atlas_path,
    atlas_info=atlas_info,
    probe_selection='diff_stability',
    donor_probes='aggregate',
    ibf_threshold=0.5,
    lr_mirror=True,
    missing='interpolate',
    tolerance=2,
    norm_matched=True,
    corrected_mni=True,
    return_donors=True,  # 需要分供体进行 DS 过滤
    return_report=True,
    verbose=2
)

# 拆包结果
donor_expression = expression_return[0]
report_text = expression_return[1]

# 保存处理方法学报告（写论文可以直接抄这里的 Method）
with open("abagen_processing_report.txt", "w") as f:
    f.write(report_text)
print("✅ Abagen 提取完成，方法学报告已保存。")

# ==========================================
# 5. DS (Differential Stability) 过滤与跨供体聚合
# ==========================================
donor_list = list(donor_expression.values())

print("正在执行差异稳定性 (DS) > 0.4 过滤...")
filtered_dfs, ds_values = keep_stable_genes(
    donor_list,
    threshold=0.4,
    return_stability=True
)

stable_gene_count = filtered_dfs[0].shape[1]
print(f"✅ 过滤完成，筛选出 DS > 0.4 的稳定基因数量: {stable_gene_count}")

# 聚合跨供体数据（求均值），得到最终的 246脑区 x N基因 的表达矩阵
print("正在聚合最终的跨供体基因表达矩阵...")
stable_expression = pd.concat(filtered_dfs).groupby(level=0).mean()

# 保存供后续 PLS 建模使用
stable_expression.to_csv(stable_matrix_path)
print(f"🎉 流程全部完成！最终基因矩阵保存至: {stable_matrix_path} | 维度: {stable_expression.shape}")