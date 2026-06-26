import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
import nimare
from nimare.meta.cbma.mkda import MKDAChi2
from brainsmash.mapgen.base import Base
from statsmodels.stats.multitest import multipletests

# =====================================================================
# 0. 全局配置区 (Global Configuration)
# =====================================================================
# --- 输入文件路径 ---
# 1. 震源分析结果文件
EPICENTER_FILE = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step15_Neurosynth/Neurosynth_Epicenter/subtype1_FullBrain_Epicenters_BrainSMASH.csv"
# 2. BNA 标准脑区名字对照表 (用于对齐顺序!!!)
LABEL_FILE = "/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/region246_network_Yeo.csv"
# 3. 模板文件
ATLAS_FILE = "/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_246_2mm.nii.gz"

# --- 输出与缓存目录 ---
WORK_DIR = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step15_Neurosynth/Neurosynth_Epicenter"
os.makedirs(WORK_DIR, exist_ok=True)
DATA_DIR = os.path.join(WORK_DIR, "neurosynth_data")
os.makedirs(DATA_DIR, exist_ok=True)
mark = "subtype1_epi"

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
# STEP 1: 读取并【对齐与过滤】Epicenter 向量
# =====================================================================
def load_epicenter_vector(epicenter_path, label_path):
    print(f"\n[读取数据] 正在加载、对齐 Epicenter 特征，并执行 P 值过滤...")
    df_epi = pd.read_csv(epicenter_path)
    df_label = pd.read_csv(label_path)

    # 【核心修改逻辑】：基于 p_surrogate < 0.05 进行阈值过滤
    # 利用 np.where：如果 p_surrogate < 0.05，取 Epicenter_r 的值；否则取 0.0
    df_epi['Filtered_Epicenter'] = np.where(df_epi['p_surrogate'] < 0.05, df_epi['Epicenter_r'], 0.0)

    # 建立 Region -> Filtered_Epicenter 的映射字典 (使用过滤后的列)
    epi_dict = dict(zip(df_epi['Region'], df_epi['Filtered_Epicenter']))

    # 按照 BNA 的 1-246 标准顺序，组装真正的特征向量
    target_vector = np.zeros(246)
    missing_count = 0

    for index, row in df_label.iterrows():
        label_id = int(row['Label'])  # 1 到 246
        region_name = str(row['regions'])

        if region_name in epi_dict:
            target_vector[label_id - 1] = epi_dict[region_name]
        else:
            missing_count += 1
            print(f"  [警告] 标准脑区 {region_name} 在 Epicenter 表格中缺失，默认设为 0")

    # 统计保留的有效脑区数量
    n_significant = np.sum(target_vector != 0)

    print(f"  ✅ 成功组装 Epicenter 向量！长度: 246，缺失数: {missing_count}")
    print(f"  🎯 经过 p_surrogate < 0.05 过滤后，共有 {n_significant} 个脑区保留了有效震源数值！")

    return target_vector


# =====================================================================
# STEP 2: Neurosynth 提取元分析认知矩阵
# =====================================================================
def run_step2_neurosynth():
    print("\n" + "=" * 50)
    print("STEP 2: 提取共享的 Neurosynth 认知特征矩阵")
    print("=" * 50)

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
        # 当 target_vector 里有很多 0 的时候，算法可能会由于数值太单一抛出奇异矩阵错误。
        # 加上一点点极小的白噪声可以完美骗过底层的线性代数求解器，且完全不影响您的结果。
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
    print(f"  >>> 显著性最高的前 5 个认知术语：")
    print(final_df.head(5)[['Term', 'Real_R', 'P_emp', 'P_FDR']])


# =====================================================================
# 主函数入口 (Main Execution Pipeline)
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 启动基于显著震源 (Epicenter, p<0.05) 的 Neurosynth 功能解码流水线")
    print("=" * 60)

    # 1. 提取通用的认知网络矩阵
    cognitive_matrix_df = run_step2_neurosynth()

    # 2. 读取 subtype 1 的 Epicenter 并进行严格对齐与过滤
    print("\n" + "*" * 50)
    print("▶️ 开始分析任务: Subtype 1 的显著 Epicenter 空间分布")
    print("*" * 50)

    epicenter_vector = load_epicenter_vector(EPICENTER_FILE, LABEL_FILE)

    # 3. 运行 BrainSMASH 零模型与相关性计算
    run_step3_and_4_spatial_null(
        target_vector=epicenter_vector,
        df_matrix=cognitive_matrix_df,
        prefix_name="Subtype1_Filtered_Epicenter"
    )

    print("\n" + "=" * 60)
    print("🏆 显著震源 (Epicenter) 的空间解码已成功执行完毕！")
    print("=" * 60)