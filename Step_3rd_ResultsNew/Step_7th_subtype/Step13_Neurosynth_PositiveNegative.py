import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
import nimare
from nimare.meta.cbma.mkda import MKDAChi2
import wget
import tarfile
from brainsmash.mapgen.base import Base
from statsmodels.stats.multitest import multipletests
'''
    此代码！
'''
# =====================================================================
# 0. 全局配置区 (Global Configuration)
# =====================================================================
# --- 输入文件路径 ---
HC_FILE = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step2_Ttest/HC_INT20_agesex.csv"
SUB1_FILE = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype1_INT_agesex.csv"
SUB2_FILE = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype2_INT_agesex.csv"  # 亚型2数据文件
ATLAS_FILE = "/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_246_2mm.nii.gz"
mark = "pos"
# --- 输出与缓存目录 ---
WORK_DIR = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step15_Neurosynth/posneg"
os.makedirs(WORK_DIR, exist_ok=True)
DATA_DIR = os.path.join(WORK_DIR, "neurosynth_data")
os.makedirs(DATA_DIR, exist_ok=True)

# 共享的 Neurosynth 矩阵保存路径
SHARED_MATRIX_OUT = os.path.join(WORK_DIR, f"{mark}_shared_neurosynth_ROI_cognitive_matrix.csv")

# --- 分析参数 ---
N_PERMUTATIONS = 10000  # 空间置换检验次数
COGNITIVE_TERMS = [
    "attention", "working memory", "executive control", "inhibitory control",
    "emotion regulation", "reward", "motivation", "fear", "anxiety", "depression",
    "episodic memory", "semantic memory", "decision making", "social cognition",
    "default mode", "salience network", "motor", "visual", "auditory", "language",
    "reading", "autobiographical memory", "reinforcement learning", "empathy",
    "pain", "arousal", "sustained attention", "cognitive control", "switching", "emotion"
]


# =====================================================================
# STEP 1: GLM 模型提取组间差异 Z-map
# =====================================================================
def run_step1_glm(hc_path, sub_path, output_csv, direction="positive"):
    """
    direction: "positive" 提取 Subtype > HC (显著且z>0)
               "negative" 提取 Subtype < HC (显著且z<0，保留原始负向z值)
    """
    print(f"\n[{direction.upper()} 分析] 正在运行 GLM 提取特征图...")

    hc_df = pd.read_csv(hc_path)
    sub_df = pd.read_csv(sub_path)
    hc_df['group'], sub_df['group'] = 0, 1
    combined_df = pd.concat([hc_df, sub_df], ignore_index=True)

    roi_cols = [col for col in hc_df.columns if col not in ['subID', 'age', 'sex', 'group']]

    X = combined_df[['group', 'age', 'sex']].copy()
    X = sm.add_constant(X)
    df_resid = len(combined_df) - X.shape[1]

    results = []
    z_vector = []
    p_vector = []

    for roi in roi_cols:
        y = combined_df[roi]
        model = sm.OLS(y, X).fit()
        t_val = model.tvalues['group']
        p_val = model.pvalues['group']
        z_score = stats.norm.ppf(stats.t.cdf(t_val, df=df_resid))

        results.append({
            'ROI': roi,
            't_value': t_val,
            'p_value': p_val,
            'z_score': z_score
        })
        z_vector.append(z_score)
        p_vector.append(p_val)

    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv, index=False)

    mdd_z_vector = np.array(z_vector)
    mdd_p_vector = np.array(p_vector)

    # ====================== 新增显著性 + 方向联合过滤 ======================
    significant = mdd_p_vector < 0.05  # 可改为 0.01 或 FDR 校正（推荐后面做FDR）

    if direction == "positive":
        target_vector = np.where(significant & (mdd_z_vector > 0), mdd_z_vector, 0)
        desc = "Subtype > HC (显著正向)"

    elif direction == "negative":
        target_vector = np.where(significant & (mdd_z_vector < 0), mdd_z_vector, 0)  # 保留原始负z值
        desc = "Subtype < HC (显著负向)"

    else:
        raise ValueError("Direction 必须是 'positive' 或 'negative'")

    n_significant = np.sum(target_vector != 0)
    print(f" ✅ GLM 计算完成！发现 {n_significant}/{len(roi_cols)} 个有效脑区: {desc}。")
    print(f"   (基于 p < 0.05 且符合方向)")

    return target_vector
# =====================================================================
# STEP 2: Neurosynth 提取元分析认知矩阵 (仅需运行一次)
# =====================================================================
def run_step2_neurosynth():
    print("\n" + "=" * 50)
    print("STEP 2: 提取共享的 Neurosynth 认知特征矩阵")
    print("=" * 50)

    # 如果本地已经有算好的矩阵，直接加载以节省大量时间
    if os.path.exists(SHARED_MATRIX_OUT):
        print(f"  ✅ 检测到本地已存在计算好的认知矩阵，直接加载：{SHARED_MATRIX_OUT}")
        return pd.read_csv(SHARED_MATRIX_OUT, index_col=0)

    try:
        files = nimare.extract.fetch_neurosynth(data_dir=DATA_DIR, version="7", source="abstract", vocab="terms",
                                                overwrite=False, return_type="files")
        ns_dataset = nimare.io.convert_neurosynth_to_dataset(coordinates_file=files[0]["coordinates"],
                                                             metadata_file=files[0]["metadata"],
                                                             annotations_files=files[0]["features"])
    except Exception as e:
        print(f"fetch 失败，尝试本地手动加载... {e}")
        ns_dataset = nimare.io.convert_neurosynth_to_dataset(
            coordinates_file=os.path.join(DATA_DIR, "data-neurosynth_version-7_coordinates.tsv.gz"),
            metadata_file=os.path.join(DATA_DIR, "data-neurosynth_version-7_metadata.tsv.gz"),
            annotations_files=[
                os.path.join(DATA_DIR, "data-neurosynth_version-7_vocabulary-terms_source-abstract.tsv.gz")]
        )

    all_columns = ns_dataset.annotations.columns.tolist()
    masker = NiftiLabelsMasker(labels_img=ATLAS_FILE, standardize=False)
    meta = MKDAChi2()
    roi_term_data = {}

    for term in COGNITIVE_TERMS:
        matched_label = next((col for col in all_columns if col.split("__")[-1].lower() == term.lower()), None)
        if not matched_label: continue
        ids_with_term = ns_dataset.get_studies_by_label(labels=matched_label, label_threshold=0.001)
        if len(ids_with_term) < 20: continue

        ids_without_term = list(set(ns_dataset.ids) - set(ids_with_term))
        try:
            meta_res = meta.fit(ns_dataset.slice(ids_with_term), ns_dataset.slice(ids_without_term))
            avail_maps = list(meta_res.maps.keys())
            map_name = "z_desc-specificity_level-voxel_stat-z" if "z_desc-specificity_level-voxel_stat-z" in avail_maps else (
                "z_2way" if "z_2way" in avail_maps else avail_maps[0])
            z_img = meta_res.get_map(map_name)
            roi_term_data[term] = masker.fit_transform(z_img).flatten()
            print(f"  ✅ 术语 [{term}] 投影成功 (纳入 {len(ids_with_term)} 篇文献)")
        except Exception as e:
            pass

    df_matrix = pd.DataFrame(roi_term_data)
    df_matrix.index = range(1, len(df_matrix) + 1)
    df_matrix.to_csv(SHARED_MATRIX_OUT)
    print(f"✅ Step 2 完成！共享矩阵维度: {df_matrix.shape}")
    return df_matrix


# =====================================================================
# STEP 3 & 4: 空间相关性与 BrainSMASH 置换检验
# =====================================================================
def extract_centroids():
    img = nib.load(ATLAS_FILE)
    data = img.get_fdata()
    affine = img.affine
    centroids = [nib.affines.apply_affine(affine, np.mean(np.array(np.where(data == roi)).T, axis=0))
                 for roi in range(1, 247)]
    return np.array(centroids)


def run_step3_and_4_spatial_null(target_vector, df_matrix, prefix_name):
    print(f"\n[{prefix_name}] 开始执行 BrainSMASH 零模型检验 ({N_PERMUTATIONS} 次)...")

    # 1. 计算真实相关系数
    real_corrs = {term: pearsonr(target_vector, df_matrix[term].values)[0] for term in df_matrix.columns}

    # 2. 准备距离矩阵
    dist_matrix = squareform(pdist(extract_centroids(), metric='euclidean'))

    # 3. BrainSMASH 生成假图
    try:
        base = Base(x=target_vector, D=dist_matrix)
    except:
        base = Base(x=target_vector + np.random.normal(0, 1e-5, len(target_vector)), D=dist_matrix)
    surrogates = base(n=N_PERMUTATIONS)

    # 4. 向量化计算置换 P 值
    results_list = []
    surr_centered = surrogates - np.mean(surrogates, axis=1, keepdims=True)
    std_surr = np.std(surrogates, axis=1) * np.sqrt(surrogates.shape[1])

    for term in df_matrix.columns:
        term_vector = df_matrix[term].values
        term_centered = term_vector - np.mean(term_vector)
        std_term = np.std(term_vector) * np.sqrt(len(term_vector))

        null_dist_r = np.dot(surr_centered, term_centered) / (std_surr * std_term)

        real_r = real_corrs[term]
        p_emp = np.sum(np.abs(null_dist_r) >= np.abs(real_r)) / N_PERMUTATIONS
        results_list.append({"Term": term, "Real_R": real_r, "P_emp": p_emp})

    # 5. FDR 多重比较校正
    final_df = pd.DataFrame(results_list)
    _, p_fdr, _, _ = multipletests(final_df['P_emp'], alpha=0.05, method='fdr_bh')
    final_df['P_FDR'] = p_fdr
    final_df['Significant_FDR'] = p_fdr < 0.05

    final_df = final_df.sort_values(by="Real_R", ascending=False).reset_index(drop=True)

    # 保存结果
    out_file = os.path.join(WORK_DIR, f"final_{prefix_name}_cognitive_p_values.csv")
    final_df.to_csv(out_file, index=False)

    print(f"  🎉 [{prefix_name}] 分析完成！结果保存至: {out_file}")
    print(f"  >>> 前 3 个显著认知术语：")
    print(final_df.head(3)[['Term', 'Real_R', 'P_emp', 'P_FDR']])


# =====================================================================
# 主函数入口 (Main Execution Pipeline)
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 启动双亚型 INT Neurosynth 功能解码流水线")
    print("=" * 60)

    # ---------------------------------------------------------
    # 【基石】先提取通用的认知网络矩阵 (提取一次，全场共享)
    # ---------------------------------------------------------
    cognitive_matrix_df = run_step2_neurosynth()

    # ---------------------------------------------------------
    # 【任务 A】分析 Subtype 1 (正向：Subtype 1 > HC)
    # ---------------------------------------------------------
    print("\n" + "*" * 50)
    print("▶️ 开始分析任务 A: Subtype 1 增加的 INT (Positive)")
    print("*" * 50)

    sub1_glm_out = os.path.join(WORK_DIR, "step1_Subtype1_vs_HC_GLM.csv")
    sub1_target_vector = run_step1_glm(HC_FILE, SUB1_FILE, sub1_glm_out, direction="positive")
    run_step3_and_4_spatial_null(sub1_target_vector, cognitive_matrix_df, prefix_name="Subtype1_Positive")

    # ---------------------------------------------------------
    # 【任务 B】分析 Subtype 2 (负向：Subtype 2 < HC)
    # ---------------------------------------------------------
    print("\n" + "*" * 50)
    print("▶️ 开始分析任务 B: Subtype 2 减少的 INT (Negative)")
    print("*" * 50)

    sub2_glm_out = os.path.join(WORK_DIR, "step1_Subtype2_vs_HC_GLM.csv")
    sub2_target_vector = run_step1_glm(HC_FILE, SUB2_FILE, sub2_glm_out, direction="negative")
    run_step3_and_4_spatial_null(sub2_target_vector, cognitive_matrix_df, prefix_name="Subtype2_Negative")

    print("\n" + "=" * 60)
    print("🏆 所有亚型的正/负向功能解码已全部成功执行！")
    print("=" * 60)