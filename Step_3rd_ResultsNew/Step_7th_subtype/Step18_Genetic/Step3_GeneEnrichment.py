import pandas as pd
import gseapy as gp
import matplotlib.pyplot as plt
import os
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# ==========================================
# Phase 3: 基因本体 (GO) 与通路 (KEGG) 富集分析
# ==========================================
print("\n=== 启动 Phase III: 基于 PLS1 核心基因的富集分析 ===")

# 直接读取上一阶段生成的 PLS 载荷结果表
final_results_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step18_Genetic/result_Gene2_Correlation/final_PLS_multi_components_loadings.csv'

try:
    results_df = pd.read_csv(final_results_path)
    print(f"✅ 成功加载关联结果表，总基因数: {len(results_df)}")
except FileNotFoundError:
    print(f"❌ 严重错误：未找到 {final_results_path}，请确保路径正确。")
    exit()

# 动态识别列名（适配你之前跑的单成分或多成分输出版本）
loading_col = 'PLS1_Loading' if 'PLS1_Loading' in results_df.columns else 'Comp1_Loading'
p_col = 'Spin_P' if 'Spin_P' in results_df.columns else 'Comp1_SpinP'

print(f"📌 使用的载荷列: [{loading_col}], P 值检验列: [{p_col}]")

# 1. 提取所有 10027 个基因作为富集的背景基因 (Background)
background_genes = results_df['Gene'].dropna().tolist()

# 2. 筛选出通过了严格空间自相关检验的显著基因 (Spin P < 0.05)
sig_df = results_df[results_df[p_col] < 0.05]

# 3. 按照对脑影像病理改变的贡献方向（正向 vs 负向），拆分为两大阵营
genes_pos = sig_df[sig_df[loading_col] > 0]['Gene'].tolist()
genes_neg = sig_df[sig_df[loading_col] < 0]['Gene'].tolist()

gene_dict = {
    'PLS1_Positive': genes_pos,  # 在萎缩严重区域【高表达】的基因
    'PLS1_Negative': genes_neg,  # 在萎缩严重区域【低表达】的基因
    'PLS1_All_Significant': sig_df['Gene'].tolist()
}

# 设定要查询的经典数据库
databases = {
    'GO_BP': 'GO_Biological_Process_2023',
    'KEGG': 'KEGG_2021_Human'
}

# 创建保存图片的文件夹
os.makedirs('Enrichment_Plots', exist_ok=True)

for group_name, gene_list in gene_dict.items():
    print(f"\n>>> 正在分析组别: {group_name} (包含 {len(gene_list)} 个显著基因)")

    # 防止基因数太少导致 API 报错或无意义
    if len(gene_list) < 10:
        print(f"⚠️ 基因数量过少 (<10)，跳过该组富集分析。")
        continue

    for db_name, db_source in databases.items():
        try:
            # 向 Enrichr 服务器发送富集请求
            enr_res = gp.enrichr(
                gene_list=gene_list,
                gene_sets=db_source,
                organism='human',
                background=background_genes,  # 严格使用当前大脑中存在的基因作为背景
                cutoff=0.05
            )

            res_df = enr_res.results
            if not res_df.empty:
                # 只保留多重比较校正后 (FDR) 依然显著的通路
                sig_res = res_df[res_df['Adjusted P-value'] < 0.05]

                if len(sig_res) > 0:
                    # 1. 保存显著通路的 CSV 报表
                    csv_filename = f"enrichment_{group_name}_{db_name}.csv"
                    sig_res.to_csv(csv_filename, index=False)
                    print(f"  ✅ {db_name} 找到 {len(sig_res)} 条显著通路，已保存报表，正在绘制气泡图...")

                    # 2. 绘制高清气泡图
                    from gseapy.plot import dotplot

                    ax = dotplot(enr_res.res2d,
                                 title=f"{db_name} Pathways\n({group_name.replace('_', ' ')})",
                                 cmap='viridis',
                                 cutoff=0.05,
                                 top_term=10)  # 仅画出最显著的前 10 条通路

                    # 提取真实画板，杜绝空白图
                    fig = ax.figure
                    fig.savefig(f"Enrichment_Plots/{group_name}_{db_name}_dotplot.png",
                                dpi=300,
                                bbox_inches='tight')
                    plt.close(fig)
                else:
                    print(f"  ⚠️ {db_name} 虽有富集，但经过 FDR 校正后无显著通路 (Adjusted P > 0.05)。")
            else:
                print(f"  ⚠️ {db_name} 未返回任何匹配通路。")

        except Exception as e:
            print(f"  ❌ 富集过程中发生网络或计算错误: {e}")

print("\n🎉 --- Phase 3 全部完成！ --- 🎉")
print("快去 Enrichment_Plots 文件夹查看最新生成的 PLS1 高清气泡图吧！")