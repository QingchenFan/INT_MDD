import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from scipy.stats import pearsonr


# 获取特征、目标变量和协变量
def get_xyc(data, ymark):
    # 提取协变量 age 和 sex
    C = data[['age', 'sex']].values  # 转为 numpy 数组，更安全

    # 提取特征 X: 从 A8m_L 开始到最后（246 个脑区）
    #start_idx = data.columns.get_loc('A8m_L')
    X = data.iloc[:, 8:].values  # 转为 numpy 数组

    # 提取目标变量 y（不做残差化）
    y = data[ymark].values

    return X, y, C


def linear_regression_cv(X, y, C=None, n_splits=5, n_runs=30, control_covariates=True):
    """
    进行交叉验证线性回归预测。
    修改后：只对脑特征 X 进行协变量残差化，不对 y 进行残差化。
    """
    all_r_scores = []
    for run in range(n_runs):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=run)  # 固定种子便于复现
        r_scores = []

        for train_index, test_index in kf.split(X):
            # 划分训练集和测试集
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]

            # ==================== 只对 X 进行协变量残差化 ====================
            if control_covariates and C is not None:
                C_train = C[train_index]
                C_test = C[test_index]

                # 1. 在训练集上拟合协变量对每个脑区的影响
                model_c_X = LinearRegression()
                model_c_X.fit(C_train, X_train)

                # 2. 计算残差（去掉协变量影响）
                X_train_final = X_train - model_c_X.predict(C_train)
                X_test_final = X_test - model_c_X.predict(C_test)
            else:
                X_train_final = X_train
                X_test_final = X_test

            # y 不做残差化，直接使用原始值
            y_train_final = y_train
            y_test_final = y_test

            # ==========================================
            # 训练主模型（这里用普通线性回归，你可以换成 RandomForestRegressor）
            # ==========================================
            model = LinearRegression()
            model.fit(X_train_final, y_train_final)

            # 预测
            y_pred = model.predict(X_test_final)

            # 计算皮尔逊相关系数
            r, _ = pearsonr(y_test_final, y_pred)
            r_scores.append(r)

        all_r_scores.append(np.mean(r_scores))

    return all_r_scores


def permutation_test(X, y, C, res, n_permutations=1000, control_covariates=False):
    """
    进行置换检验计算 p 值。
    注意：置换只打乱 y，X 和协变量处理方式与主实验保持一致。
    """

    permutation_r_values = []
    for _ in range(n_permutations):
        # 随机打乱目标变量 y
        y_permuted = np.random.permutation(y)

        # 对置换数据进行一次交叉验证
        permuted_r_list = linear_regression_cv(
            X, y_permuted, C,
            n_splits=5,  # 与主实验一致
            n_runs=1,
            control_covariates=control_covariates
        )
        permutation_r_values.append(permuted_r_list[0])

    # 计算单侧 p 值（真实 r 越大越显著）
    p_value = np.mean(np.array(permutation_r_values) >= res)
    return p_value


# --- 主执行程序 ---
if __name__ == '__main__':
    file_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZ/Step4_scale/INT_MDD_HAMD.csv'
    data = pd.read_csv(file_path)

    # 提取数据（现在 X 和 C 已经是 numpy 数组）
    X, y, C = get_xyc(data, 'HAMD_response')
    print('SHAPE:', 'X:', X.shape, 'y:', y.shape, 'Covariates:', C.shape)

    # ==========================================
    # 个性化设置区
    # ==========================================
    USE_COVARIATES = False  # 改为 False 即可运行【不控制】协变量的版本
    # ==========================================

    if USE_COVARIATES:
        print("\n--- 当前模式：控制协变量 (Age, Sex) --- 只对脑特征 X 进行残差化 ---")
    else:
        print("\n--- 当前模式：不控制协变量 (直接预测) ---")

    # 1. 真实模型评估
    all_r_scores = linear_regression_cv(X, y, C,
                                        n_splits=5,
                                        n_runs=30,
                                        control_covariates=USE_COVARIATES)
    res = np.median(all_r_scores)
    print(f'真实预测 r 值 (Median over {len(all_r_scores)} runs): {res:.4f}')

    # 2. 置换检验（测试时可先设 n_permutations=200）
    pvalue = permutation_test(X, y, C, res,
                              n_permutations=1000,
                              control_covariates=USE_COVARIATES)
    print(f'置换检验 P-value: {pvalue:.4f}')