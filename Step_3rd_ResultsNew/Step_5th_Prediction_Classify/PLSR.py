import os
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr


# 获取特征、目标变量和协变量
def get_xyc(data, ymark):
    C = data[['age', 'sex']].values
    X = data.iloc[:, 6:].values  # 从 A8m_L 开始的 246 个脑区
    y = data[ymark].values

    return X, y, C


def plsr_cv(X, y, C=None, n_splits=5, n_runs=50, control_covariates=True, save_results=False,
            save_dir='./plsr_results'):
    """
    使用 PLSRegression 进行交叉验证预测。
    只对脑特征 X 进行协变量残差化，不对 y 进行残差化。
    """
    all_r_scores = []

    # 如果需要保存结果，先创建主目录
    if save_results and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for run in range(n_runs):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=run)
        r_scores = []

        # 为当前 run 创建专属文件夹
        if save_results:
            run_dir = os.path.join(save_dir, f'run_{run}')
            os.makedirs(run_dir, exist_ok=True)

        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            # 划分数据集
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]

            C_train = C[train_index] if C is not None else None
            C_test = C[test_index] if C is not None else None

            # ==================== 只对 X 进行协变量残差化 ====================
            if control_covariates and C is not None:
                model_c_X = LinearRegression()
                model_c_X.fit(C_train, X_train)
                X_train_final = X_train - model_c_X.predict(C_train)
                X_test_final = X_test - model_c_X.predict(C_test)
            else:
                X_train_final = X_train.copy()
                X_test_final = X_test.copy()

            # y 不做残差化（保持原始值）
            y_train_final = y_train
            y_test_final = y_test

            # ==================== 特征标准化（PLSR 强烈推荐） ====================
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_final)
            X_test_scaled = scaler.transform(X_test_final)

            # ==========================================
            # 主模型：PLSR (Partial Least Squares Regression)
            # ==========================================
            model = PLSRegression(n_components=2)  # 关键参数！建议 5~20 之间
            # y 需要 reshape 成二维 (n_samples, 1)
            model.fit(X_train_scaled, y_train_final.reshape(-1, 1))

            # 预测并压平
            y_pred = model.predict(X_test_scaled).ravel()

            # 计算皮尔逊相关系数
            r, _ = pearsonr(y_test_final, y_pred)
            r_scores.append(r)

            # ==================== 保存每一折的结果 ====================
            if save_results:
                # 1. 保存预测值和真实值
                df_pred = pd.DataFrame({
                    'y_true': y_test_final,
                    'y_pred': y_pred
                })
                pred_path = os.path.join(run_dir, f'fold_{fold}_predictions.csv')
                df_pred.to_csv(pred_path, index=False)

                # 2. 保存模型权重 (PLSR 的 coef_ 形状为 (n_features, n_targets))
                weights = model.coef_

                df_weights = pd.DataFrame(weights[0], columns=['weight'])
                weights_path = os.path.join(run_dir, f'fold_{fold}_weights.csv')
                df_weights.to_csv(weights_path, index=False)

        all_r_scores.append(np.mean(r_scores))

    return all_r_scores


def permutation_test(X, y, C, res, n_permutations=1000, control_covariates=True):
    """置换检验（只打乱 y）"""
    permutation_r_values = []
    for _ in range(n_permutations):
        y_permuted = np.random.permutation(y)

        # 注意：置换检验时务必关闭 save_results，避免生成海量无用文件
        permuted_r_list = plsr_cv(
            X, y_permuted, C,
            n_splits=5,
            n_runs=1,
            control_covariates=control_covariates,
            save_results=False
        )
        permutation_r_values.append(permuted_r_list[0])

    p_value = np.mean(np.array(permutation_r_values) >= res)
    return p_value


# --- 主执行程序 ---
if __name__ == '__main__':
    file_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step5_HAMD_Prediction/subtype2_INT246_HAMD.csv'
    data = pd.read_csv(file_path)

    X, y, C = get_xyc(data, 'HAMD_response')
    print('SHAPE:', 'X:', X.shape, 'y:', y.shape, 'Covariates:', C.shape)

    # ==========================================
    # 个性化设置区
    # ==========================================
    USE_COVARIATES = True  # True: 控制协变量 | False: 不控制
    SAVE_RESULTS = True  # True: 开启保存 | False: 不保存
    SAVE_DIR = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step5_HAMD_Prediction/PLSR'  # 保存结果的根目录
    # ==========================================

    if USE_COVARIATES:
        print("\n--- 当前模式：控制协变量 (Age, Sex) --- 只对脑特征 X 进行残差化 ---")
    else:
        print("\n--- 当前模式：不控制协变量 (直接预测) ---")

    # 1. 真实模型评估 (使用 PLSR)
    print("\n[正在运行真实模型并保存结果...]")
    all_r_scores = plsr_cv(X, y, C,
                           n_splits=5,
                           n_runs=101,  # 建议 30~100
                           control_covariates=USE_COVARIATES,
                           save_results=SAVE_RESULTS,
                           save_dir=SAVE_DIR)

    # 计算统计信息
    res_mean = np.mean(all_r_scores)
    res_median = np.median(all_r_scores)
    res_std = np.std(all_r_scores)

    # 找到 101 次实验中，处于中位数位置的索引
    # 使用 argsort 对数组排序，取中间那个元素的原始索引即可代表中位数对应的 run_index
    sorted_indices = np.argsort(all_r_scores)
    median_run_index = sorted_indices[len(all_r_scores) // 2]

    print("\n=== 模型评估结果 ===")
    print(f'所有 {len(all_r_scores)} 次重复 CV 的平均 r 值 (Mean): {res_mean:.4f}')
    print(f'中位数 r 值 (Median): {res_median: .4f}')
    print(f'标准差 (Std): {res_std: .4f}')
    print(
        f'>>> 【重要】中位数 r 值对应的实验轮次索引 (Run Index) 是: run_{median_run_index} (值为: {all_r_scores[median_run_index]: .4f}) <<<')

    # 2. 置换检验
    print("\n[正在运行置换检验...]")
    pvalue = permutation_test(X, y, C, res_mean,
                              n_permutations=1000,
                              control_covariates=USE_COVARIATES)
    print(f'\n置换检验 P-value: {pvalue: .4f}')