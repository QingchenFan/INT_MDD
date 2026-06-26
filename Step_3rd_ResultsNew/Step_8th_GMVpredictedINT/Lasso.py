import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# 1. 读取数据
file_name = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step14_GMVpredictedINT/subtype1_GMV246_7NetINT.csv'
df = pd.read_csv(file_name)

# 2. 精确提取特征列（从 “A8m_R” 到 “lPFtha_R”）
cols = df.columns.tolist()
start_idx = cols.index('A8m_R')
end_idx = cols.index('lPFtha_R')
feature_cols = cols[start_idx:end_idx + 1]

# 3. 明确 8 个网络的目标列
target_cols = [
    'subcortical', 'Visual', 'Somatomotor', 'Dorsal_Attention',
    'Ventral_Attention', 'Limbic', 'Frontoparietal', 'Default'
]

X = df[feature_cols].values
targets = df[target_cols].values

# 4. 设置十折交叉验证 (10-Fold CV)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 初始化用于存放每个被试交叉验证预测值的矩阵
oof_preds = np.zeros(targets.shape)
r2_scores = []

# 5. 循环单独对 8 个网络进行训练与预测（共运算 8 次）
print("开始进行十折交叉验证 Lasso 回归预测...")
for i, target_name in enumerate(target_cols):
    y = targets[:, i]
    y_pred_all = np.zeros(len(y))

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 【核心修改与优化】
        # 1. 使用 make_pipeline 确保先在训练集内部进行标准化(StandardScaler)，再训练Lasso，彻底避免数据泄露
        # 2. 使用 LassoCV 自动寻找最佳的正则化系数 alpha（自动在 log 空间生成 100 个 alpha 进行筛选）
        # 3. max_iter=10000 确保 Lasso 优化算法完全收敛
        model = make_pipeline(
            StandardScaler(),
            LassoCV(alphas=np.logspace(-4, 1, 100), max_iter=10000, cv=5, random_state=42)
        )
        model.fit(X_train, y_train)

        # 存放测试集的预测结果
        y_pred_all[test_idx] = model.predict(X_test)

    # 保存该网络的 Out-of-fold 预测值
    oof_preds[:, i] = y_pred_all

    # 使用计算得到的全样本预测值与真实值计算该网络的总体 R2
    r2 = r2_score(y, y_pred_all)
    r2_scores.append(r2)
    print(f"网络: {target_name:<18} | 十折交叉验证 R² = {r2:.4f}")

# 6. 计算 8 个网络的平均 R2 (对应 Fig.3C 的柱子高度)
mean_r2 = np.mean(r2_scores)
print(f"\n>>>> 8个网络的平均预测 R² (Mean R²) = {mean_r2:.4f} <<<<\n")


# =====================================================================
# 7. 保存结果到 CSV
# =====================================================================

# 【方式 A】仅保存 8 个网络的 R2 评分以及它们的平均 R2（1行指标数据）
r2_data = {'subID': ['R2']}
for i, target_name in enumerate(target_cols):
    r2_data[target_name] = [r2_scores[i]]

# 追加一列用于存放 8 个网络的平均 R2
r2_data['Mean_R2'] = [mean_r2]

df_r2_only = pd.DataFrame(r2_data)
output_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step14_GMVpredictedINT/lasso_s1_r2_results.csv'
df_r2_only.to_csv(output_path, index=False)
print(f"[成功] Lasso 模型的指标汇总结果已保存至 '{output_path}'")