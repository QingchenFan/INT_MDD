import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression   # Ridge 用于主模型
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr


# 获取特征、目标变量和协变量
def get_xyc(data, ymark):
    C = data[['age', 'sex']].values
    X = data.iloc[:, 8:].values          # 从 A8m_L 开始的 246 个脑区
    y = data[ymark].values
    return X, y, C


def ridge_cv(X, y, C=None, n_splits=5, n_runs=50, control_covariates=True):
    """
    使用 Ridge（岭回归）进行交叉验证预测。
    只对脑特征 X 进行协变量残差化，不对 y 进行残差化。
    """
    all_r_scores = []

    for run in range(n_runs):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=run)
        r_scores = []

        for train_index, test_index in kf.split(X):
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

            # y 不做残差化
            y_train_final = y_train
            y_test_final = y_test

            # ==================== 特征标准化 ====================
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_final)
            X_test_scaled = scaler.transform(X_test_final)

            # ==========================================
            # 主模型：Ridge 岭回归
            # ==========================================
            model = Ridge(alpha=10.0, random_state=run)   # alpha 可调（越大正则化越强）
            model.fit(X_train_scaled, y_train_final)

            # 预测
            y_pred = model.predict(X_test_scaled)

            # 计算皮尔逊相关系数
            r, _ = pearsonr(y_test_final, y_pred)
            r_scores.append(r)

        all_r_scores.append(np.mean(r_scores))

    return all_r_scores


def permutation_test(X, y, C, res, n_permutations=1000, control_covariates=True):
    """置换检验（只打乱 y）"""
    permutation_r_values = []
    for _ in range(n_permutations):
        y_permuted = np.random.permutation(y)

        permuted_r_list = ridge_cv(
            X, y_permuted, C,
            n_splits=5,
            n_runs=1,
            control_covariates=control_covariates
        )
        permutation_r_values.append(permuted_r_list[0])

    p_value = np.mean(np.array(permutation_r_values) >= res)
    return p_value


# --- 主执行程序 ---
if __name__ == '__main__':
    file_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZ/Step4_scale/INT_MDD_HAMD.csv'
    data = pd.read_csv(file_path)

    X, y, C = get_xyc(data, 'HAMD_response')
    print('SHAPE:', 'X:', X.shape, 'y:', y.shape, 'Covariates:', C.shape)

    # ==========================================
    # 个性化设置区
    # ==========================================
    USE_COVARIATES = False   # True: 控制协变量 | False: 不控制
    # ==========================================

    if USE_COVARIATES:
        print("\n--- 当前模式：控制协变量 (Age, Sex) --- 只对脑特征 X 进行残差化 ---")
    else:
        print("\n--- 当前模式：不控制协变量 (直接预测) ---")

    # 1. 真实模型评估 (使用 Ridge)
    all_r_scores = ridge_cv(X, y, C,
                            n_splits=5,
                            n_runs=50,           # 建议至少 30~50
                            control_covariates=USE_COVARIATES)

    # 使用平均值（推荐） + 打印更多统计信息
    res_mean = np.mean(all_r_scores)
    res_median = np.median(all_r_scores)
    res_std = np.std(all_r_scores)

    print(f'所有 {len(all_r_scores)} 次重复 CV 的平均 r 值 (Mean): {res_mean:.4f}')
    print(f'中位数 r 值 (Median): {res_median:.4f}')
    print(f'标准差 (Std): {res_std:.4f}')

    # 2. 置换检验
    pvalue = permutation_test(X, y, C, res_mean,   # 注意这里传入 mean
                              n_permutations=1000,
                              control_covariates=USE_COVARIATES)
    print(f'置换检验 P-value: {pvalue:.4f}')