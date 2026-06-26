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
from brainsmash.mapgen.base import Base
from statsmodels.stats.multitest import multipletests
import warnings

warnings.filterwarnings("ignore")

# =====================================================================
# 0. 全局配置区 (Global Configuration) - 【每次只需在这里换路径！】
# =====================================================================
# --- 1. 数据输入路径 ---
HC_FILE = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step2_Ttest/HC_INT20_agesex.csv"

# 🎯 【核心切换区】：要跑亚型1，就填亚型1的路径；要跑亚型2，就换成亚型2的路径！
#TARGET_SUBTYPE_FILE = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype1_INT_agesex.csv"
TARGET_SUBTYPE_FILE = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype2_INT_agesex.csv"

ATLAS_FILE = "/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_246_2mm.nii.gz"

# --- 2. 给本次运行起个名字（用于区分输出文件） ---
ANALYSIS_NAME = "Subtype2_Continuous"  # 如果跑亚型2，可以改成 "Subtype2_Continuous"

# --- 3. 输出与缓存目录 ---
WORK_DIR = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step15_Neurosynth_Continuous"
os.makedirs(WORK_DIR, exist_ok=True)
DATA_DIR = os.path.join(WORK_DIR, "neurosynth_data")
os.makedirs(DATA_DIR, exist_ok=True)

SHARED_MATRIX_OUT = os.path.join(WORK_DIR, f"{ANALYSIS_NAME}shared_neurosynth_ROI_cognitive_matrix.csv")

# --- 4. 分析参数 ---
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
# STEP 1: GLM 提取连续全脑特征 (不设阈值，保留完整 Z 分数)
# =====================================================================
def run_step1_glm_continuous(hc_path, sub_path, out_prefix):
    print(f"\n[STEP 1] 正在运行 GLM 提取全脑连续异常模式 ({out_prefix})...")

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

    for roi in roi_cols:
        y = combined_df[roi]
        model = sm.OLS(y, X).fit()
        t_val = model.tvalues['group']
        z_score = stats.norm.ppf(stats.t.cdf(t_val, df=df_resid))
        results.append({'ROI': roi, 't_value': t_val, 'z_score': z_score})
        z_vector.append(z_score)

    res_df = pd.DataFrame(results)
    out_csv = os.path.join(WORK_DIR, f"{out_prefix}_GLM_results.csv")
    res_df.to_csv(out_csv, index=False)

    mdd_z_vector = np.array(z_vector)
    return mdd_z_vector


# =====================================================================
# STEP 2: Neurosynth 提取元分析认知矩阵 (共享资源，跑一次即可)
# =====================================================================

def run_step2_neurosynth():
    print("\n[STEP 2] 提取 Neurosynth 认知特征矩阵...")
    if os.path.exists(SHARED_MATRIX_OUT):
        print(f"  ✅ 检测到本地矩阵缓存，直接加载，极速跳过！")
        return pd.read_csv(SHARED_MATRIX_OUT, index_col=0)

    # 声明 NiMARE 实际存放数据的子目录路径
    NEUROSYNTH_DIR = os.path.join(DATA_DIR, "neurosynth")

    try:
        files = nimare.extract.fetch_neurosynth(data_dir=DATA_DIR, version="7", source="abstract", vocab="terms",
                                                overwrite=False, return_type="files")
        ns_dataset = nimare.io.convert_neurosynth_to_dataset(coordinates_file=files[0]["coordinates"],
                                                             metadata_file=files[0]["metadata"],
                                                             annotations_files=files[0]["features"])
    except Exception as e:
        print(f"  [Info] Try 块加载失败 ({e})，正在尝试使用修正后的绝对路径读取本地文件...")
        # 在这里补上 NEUROSYNTH_DIR 层级
        ns_dataset = nimare.io.convert_neurosynth_to_dataset(
            coordinates_file=os.path.join(NEUROSYNTH_DIR, "data-neurosynth_version-7_coordinates.tsv.gz"),
            metadata_file=os.path.join(NEUROSYNTH_DIR, "data-neurosynth_version-7_metadata.tsv.gz"),
            annotations_files=[
                os.path.join(NEUROSYNTH_DIR, "data-neurosynth_version-7_vocabulary-terms_source-abstract.tsv.gz")]
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
            print(f"  ✅ 术语 [{term}] 投影成功")
        except Exception:
            pass

    df_matrix = pd.DataFrame(roi_term_data)
    df_matrix.index = range(1, len(df_matrix) + 1)
    df_matrix.to_csv(SHARED_MATRIX_OUT)
    return df_matrix


# =====================================================================
# STEP 3 & 4: 全脑连续空间相关性与 BrainSMASH 置换检验
# =====================================================================
def extract_centroids():
    img = nib.load(ATLAS_FILE)
    data = img.get_fdata()
    affine = img.affine
    return np.array(
        [nib.affines.apply_affine(affine, np.mean(np.array(np.where(data == roi)).T, axis=0)) for roi in range(1, 247)])


def run_step3_and_4_spatial_null_continuous(target_vector, df_matrix, out_prefix):
    print(f"\n[STEP 3 & 4] 开始执行全脑连续关联的 BrainSMASH 零模型检验 ({N_PERMUTATIONS} 次)...")

    # 1. 计算全脑连续相关系数 (Pearson r)
    real_corrs = {term: pearsonr(target_vector, df_matrix[term].values)[0] for term in df_matrix.columns}

    # 2. 准备距离矩阵
    dist_matrix = squareform(pdist(extract_centroids(), metric='euclidean'))

    # 3. BrainSMASH 生成保留全脑平滑地形图的假图
    try:
        base = Base(x=target_vector, D=dist_matrix)
    except:
        base = Base(x=target_vector + np.random.normal(0, 1e-5, len(target_vector)), D=dist_matrix)
    surrogates = base(n=N_PERMUTATIONS)

    # 4. 向量化极速计算置换 P 值
    results_list = []
    surr_centered = surrogates - np.mean(surrogates, axis=1, keepdims=True)
    std_surr = np.std(surrogates, axis=1) * np.sqrt(surrogates.shape[1])

    for term in df_matrix.columns:
        term_vector = df_matrix[term].values
        term_centered = term_vector - np.mean(term_vector)
        std_term = np.std(term_vector) * np.sqrt(len(term_vector))

        null_dist_r = np.dot(surr_centered, term_centered) / (std_surr * std_term)

        real_r = real_corrs[term]
        # Empirical P-value (双侧检验)
        p_emp = np.sum(np.abs(null_dist_r) >= np.abs(real_r)) / N_PERMUTATIONS
        results_list.append({"Term": term, "Real_R": real_r, "P_emp": p_emp})

    # 5. FDR 多重比较校正
    final_df = pd.DataFrame(results_list)
    _, p_fdr, _, _ = multipletests(final_df['P_emp'], alpha=0.05, method='fdr_bh')
    final_df['P_FDR'] = p_fdr
    final_df['Significant_FDR'] = p_fdr < 0.05

    # 【解释引导列】：为了方便您在论文里写结论，代码自动帮您判定 R 的正负含义！
    final_df['Biological_Meaning'] = final_df['Real_R'].apply(
        lambda r: "网络落于 INT 延长区" if r > 0 else "网络落于 INT 缩短区"
    )

    # 按显著性排序
    final_df = final_df.sort_values(by="P_emp", ascending=True).reset_index(drop=True)

    out_file = os.path.join(WORK_DIR, f"final_{out_prefix}_continuous_p_values.csv")
    final_df.to_csv(out_file, index=False)

    print(f"  🎉 分析竣工！结果已保存至: {out_file}")
    print(f"\n>>> 【{out_prefix}】最显著的前 5 个全脑空间关联结果：")
    print(final_df.head(5)[['Term', 'Real_R', 'P_emp', 'P_FDR', 'Biological_Meaning']])


# =====================================================================
# 主函数 (运行入口)
# =====================================================================
if __name__ == "__main__":
    print(f"\n" + "=" * 70)
    print(f"🚀 启动【全脑连续模式法】Neurosynth 功能解码 - 当前任务: {ANALYSIS_NAME}")
    print("=" * 70)

    # 1. 提取全脑连续特征图 (包含正值和负值的完整 246 Z 分数向量)
    continuous_disease_vector = run_step1_glm_continuous(HC_FILE, TARGET_SUBTYPE_FILE, out_prefix=ANALYSIS_NAME)

    # 2. 提取认知特征矩阵 (自动使用缓存)
    cognitive_matrix_df = run_step2_neurosynth()

    # 3. 计算全脑连续关联与 BrainSMASH 置换检验
    run_step3_and_4_spatial_null_continuous(continuous_disease_vector, cognitive_matrix_df, out_prefix=ANALYSIS_NAME)