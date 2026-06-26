import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


def partial_correlation_analysis(df_path, var_x, var_y, covariates, alpha=0.05):
    """
    计算两个变量在控制了指定协变量后的偏相关系数（Pearson & Spearman）。
    如果在任一检验中显著，则生成两张散点图：一张展示 Pearson 统计量，一张展示 Spearman 统计量。

    参数:
    - df_path (str): CSV 文件路径
    - var_x (str): 自变量/特征名称 (X)
    - var_y (str): 因变量/标签名称 (Y)
    - covariates (list): 协变量列表，例如 ['age', 'sex', 'mean_fd']
    - alpha (float): 显著性水平，默认 0.05
    """

    # 1. 读取数据
    df = pd.read_csv(df_path)

    # 2. 提取涉及到的所有列，并去除包含缺失值的行，确保回归模型正常运行
    cols = [var_x, var_y] + covariates
    df = df.dropna(subset=cols)

    # 定义函数：通过线性回归计算剔除协变量影响后的残差
    def get_residuals(target, covars, data):
        X = data[covars]
        X = sm.add_constant(X)  # 添加截距项
        y = data[target]
        model = sm.OLS(y, X).fit()
        return model.resid

    # 3. 分别回归掉协变量的影响，得到 var_x 和 var_y 的残差序列
    res_x = get_residuals(var_x, covariates, df)
    res_y = get_residuals(var_y, covariates, df)

    # 4. 计算残差之间的相关性 (即偏相关)
    r_pearson, p_pearson = stats.pearsonr(res_x, res_y)
    r_spearman, p_spearman = stats.spearmanr(res_x, res_y)

    print(f"========== 偏相关分析结果 ==========")
    print(f"X: {var_x}")
    print(f"Y: {var_y}")
    print(f"控制变量: {covariates}")
    print("-" * 35)
    print(f"Pearson  偏相关: r = {r_pearson:.4f}, p = {p_pearson:.4f}")
    print(f"Spearman 偏相关: r = {r_spearman:.4f}, p = {p_spearman:.4f}")
    print("====================================")

    # 5. 检查是否显著，如果任意一个显著则画两张图
    if p_pearson < alpha or p_spearman < alpha:
        print("\n=> 发现显著相关 (p < 0.05)！正在生成两张偏相关散点图 (Pearson & Spearman)...")

        # 将两种统计量打包，方便循环绘图
        plot_configs = [
            {'method': 'Pearson', 'r': r_pearson, 'p': p_pearson},
            {'method': 'Spearman', 'r': r_spearman, 'p': p_spearman}
        ]

        for config in plot_configs:
            method = config['method']
            r_val = config['r']
            p_val = config['p']

            plt.figure(figsize=(8, 6))
            # 绘制散点和回归线
            sns.regplot(x=res_x, y=res_y,
                        scatter_kws={'alpha': 0.6, 'color': '#2c7fb8'},
                        line_kws={'color': '#d95f02'})

            # 构建要显示的文本字符串
            p_text = f"p < 0.0001" if p_val < 0.0001 else f"p = {p_val:.4f}"
            stats_text = f"{method} r = {r_val:.4f}\n{p_text}"

            # 使用 plt.text 将文本放置在图表的左上角
            plt.text(0.05, 0.95, stats_text,
                     transform=plt.gca().transAxes,
                     fontsize=12,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

            # 坐标轴加上残差标识，标明这是控制了协变量的图
            plt.xlabel(f"{var_x} (Residuals after controlling covariates)")
            plt.ylabel(f"{var_y} (Residuals after controlling covariates)")
            plt.title(f"Partial Correlation ({method}): {var_x} vs {var_y}\n(Covariates: {', '.join(covariates)})")
            plt.grid(True, linestyle='--', alpha=0.5)

            # 保存图片，文件名中加上方法名以作区分
            plot_filename = f'partial_corr_{method.lower()}_{var_x}_{var_y}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  -> 绘图完成并保存为: {plot_filename}")
    else:
        print("\n=> 相关性不显著 (p >= 0.05)，不生成散点图。")

    return {
        'Pearson_r': r_pearson, 'Pearson_p': p_pearson,
        'Spearman_r': r_spearman, 'Spearman_p': p_spearman
    }


# ================= 测试调用示例 =================
if __name__ == "__main__":
    path = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_INTSlope/Slope_Scale_Corr/subtype1_slope_HCL.csv"


    res = partial_correlation_analysis(
        df_path=path,
        var_x='slope',
        var_y='HCL_sum',
        covariates=['age', 'sex', 'mean_fd'],
        alpha=0.05
    )