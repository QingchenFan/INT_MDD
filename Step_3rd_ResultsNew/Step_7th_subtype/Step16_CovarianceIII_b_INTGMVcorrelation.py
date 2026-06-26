import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. 核心统计函数：Mantel Test (矩阵相关性置换检验)
# ==========================================
def mantel_test(mat1, mat2, permutations=5000):
    """
    对两个对称矩阵执行 Mantel Test，解决边不独立导致的自由度膨胀问题。
    """
    n = mat1.shape[0]
    upper_tri = np.triu_indices(n, k=1)

    # 提取上三角并计算真实的皮尔逊相关系数
    vec1 = mat1[upper_tri]
    vec2 = mat2[upper_tri]
    true_r, _ = stats.pearsonr(vec1, vec2)

    # 初始化空数组存放假 r 值
    null_dist = np.zeros(permutations)

    # 开始置换循环
    for i in range(permutations):
        # 随机打乱脑区 (1~246) 的顺序
        perm = np.random.permutation(n)

        # 对矩阵1的行和列同时进行相同的打乱，保持网络内在拓扑结构不被破坏
        mat1_permuted = mat1[perm, :][:, perm]

        # 提取打乱后的上三角，与原矩阵2计算相关性
        vec1_permuted = mat1_permuted[upper_tri]
        null_r, _ = stats.pearsonr(vec1_permuted, vec2)

        null_dist[i] = null_r

    # 计算非参数 P 值：假分布中绝对值大于等于真实 r 值的比例
    # (加 1 是为了避免 p=0，符合置换检验规范)
    p_value = (np.sum(np.abs(null_dist) >= np.abs(true_r)) + 1) / (permutations + 1)

    return true_r, p_value, null_dist


# ==========================================
# 2. 执行分析与绘图
# ==========================================
def analyze_network_coupling(gmv_path, int_path, output_name, subtype_name):
    print(f"\n开始分析 {subtype_name} ...")

    # 读取相关性矩阵 (已经是 r 值)
    gmv_df = pd.read_csv(gmv_path, index_col=0)
    int_df = pd.read_csv(int_path, index_col=0)

    # 转换为 numpy 数组，并执行 Fisher Z 转换
    # (这是必须要做的数值分布变换)
    gmv_z_mat = np.arctanh(np.clip(gmv_df.values, -0.9999, 0.9999))
    int_z_mat = np.arctanh(np.clip(int_df.values, -0.9999, 0.9999))

    print("正在执行 5000 次 Mantel Test 置换检验，请稍候...")
    # 调用 Mantel Test
    r_val, p_val, _ = mantel_test(gmv_z_mat, int_z_mat, permutations=5000)

    # 格式化 P 值
    p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"

    # 提取绘图用的散点数据
    upper_tri = np.triu_indices_from(gmv_z_mat, k=1)
    gmv_vec = gmv_z_mat[upper_tri]
    int_vec = int_z_mat[upper_tri]

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    sns.regplot(x=gmv_vec, y=int_vec,
                scatter_kws={'alpha': 0.1, 's': 2, 'color': '#2ca02c'},
                line_kws={'color': 'darkred', 'linewidth': 2}, ax=ax)

    # 显示结果
    textstr = f'Mantel $r = {r_val:.3f}$\nMantel {p_str}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.title(f'Micro-Macro Connectome Coupling ({subtype_name})\nGMV vs INT Covariance Networks', fontsize=14, pad=15)
    plt.xlabel('GMV Covariance (Fisher Z)', fontsize=12)
    plt.ylabel('INT Covariance (Fisher Z)', fontsize=12)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_name, dpi=300, facecolor='white')
    plt.close()

    print(f"[{subtype_name}] 分析完成！Mantel r = {r_val:.4f}, Mantel p = {p_val:.4f}")


# ==========================================
# 3. 运行批处理
# ==========================================
base_dir = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step17_CovariationGMV/result1_INTGMVCorrelation'

analyze_network_coupling(
    gmv_path=f'{base_dir}/subtype1_GMV_covariance_network_r.csv',
    int_path=f'{base_dir}/subtype1_INT_covariance_network_r.csv',
    output_name=f'{base_dir}/Mantel_Correlation_Subtype1.png',
    subtype_name='Subtype 1'
)

analyze_network_coupling(
    gmv_path=f'{base_dir}/subtype2_GMV_covariance_network_r.csv',
    int_path=f'{base_dir}/subtype2_INT_covariance_network_r.csv',
    output_name=f'{base_dir}/Mantel_Correlation_Subtype2.png',
    subtype_name='Subtype 2'
)