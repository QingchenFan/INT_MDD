import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score

# 1. 读取数据
data_path = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_NetworkHierarchy/INT_DiffNetwork/DiffNetwork_FirstEp_ classify/subtype2_DiffNetwork_FirstEp.csv"
df = pd.read_csv(data_path)

# 2. 提取目标变量、协变量和脑网络特征
y = df['FirstEpisode'].values
# 将标签 1 和 2 映射为 0 和 1（SVC分类器处理二分类的标准格式）
if set(y) == {1, 2}:
    y = y - 1

# 协变量：包含年龄、性别，以及平均头动参数 mean_fd
covariates = df[['age', 'sex', 'mean_fd']].values

# 脑网络特征：从 'Default-Frontoparietal' 一直到最后一列 'Visual-subcortical'
start_idx = df.columns.get_loc('Default-Frontoparietal')
X_networks = df.iloc[:, start_idx:].values

print(f"样本量: {X_networks.shape[0]}, 脑网络特征维度: {X_networks.shape[1]}")


# 3. 定义核心评测函数（Fold内协变量校正 + 标准化 + SVM分类）
def evaluate_svm_pipeline(X_net, cov, y_true, cv, kernel='sigmoid', C=1.0):
    scores_acc = []
    scores_auc = []

    for train_idx, test_idx in cv.split(X_net, y_true):
        # 划分训练集和测试集
        X_train_net, X_test_net = X_net[train_idx], X_net[test_idx]
        cov_train, cov_test = cov[train_idx], cov[test_idx]
        y_train, y_test = y_true[train_idx], y_true[test_idx]

        # --- 步骤 A: 协变量残差化 (严格在Fold内部进行，严防数据泄露) ---
        reg = LinearRegression()
        reg.fit(cov_train, X_train_net)

        # 计算残差作为除去协变量影响后的干净特征
        X_train_resid = X_train_net - reg.predict(cov_train)
        X_test_resid = X_test_net - reg.predict(cov_test)

        # --- 步骤 B: 特征标准化 ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_resid)
        X_test_scaled = scaler.transform(X_test_resid)

        # --- 步骤 C: SVM分类器拟合与预测 ---
        # 影像数据推荐优先使用线性核('linear')；C为惩罚系数
        # 注意：必须设置 probability=True 才能计算概率并输出 AUC 指标
        clf = SVC(kernel=kernel, C=C, probability=True, random_state=42)
        clf.fit(X_train_scaled, y_train)

        # 预测结果
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]

        scores_acc.append(accuracy_score(y_test, y_pred))
        scores_auc.append(roc_auc_score(y_test, y_prob))

    return np.mean(scores_acc), np.mean(scores_auc)


# 4. 运行真实的 5 折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# 这里先以线性核、C=1.0 为默认参数运行
real_acc, real_auc = evaluate_svm_pipeline(X_networks, covariates, y, cv, kernel='linear', C=1.0)

print("\n=== SVM 真实模型评估结果 ===")
print(f"5折交叉验证平均准确率 (Accuracy): {real_acc:.4f}")
print(f"5折交叉验证平均曲线下面积 (AUC): {real_auc:.4f}")

# 5. 置换检验 (Permutation Testing)
print("\n正在进行置换检验，请稍候...")
n_permutations = 1000  # 学术发表标准通常为 1000 或 10000 次
null_accs = []
null_aucs = []

for i in range(n_permutations):
    # 仅随机打乱疾病标签，保持特征和协变量的配对关系
    y_permuted = np.random.permutation(y)

    # 运行与真实模型完全相同的 Pipeline
    perm_acc, perm_auc = evaluate_svm_pipeline(X_networks, covariates, y_permuted, cv, kernel='linear', C=1.0)
    null_accs.append(perm_acc)
    null_aucs.append(perm_auc)

    if (i + 1) % 200 == 0:
        print(f"已完成 {i + 1} / {n_permutations} 次置换")

# 计算非参数 p 值 (观察真实得分在零分布中的排位)
p_acc = (np.sum(np.array(null_accs) >= real_acc) + 1) / (n_permutations + 1)
p_auc = (np.sum(np.array(null_aucs) >= real_auc) + 1) / (n_permutations + 1)

print("\n=== 置换检验统计学显著性结果 ===")
print(f"Accuracy 置换检验 p 值: {p_acc:.4f} " + ("(统计学显著)" if p_acc < 0.05 else "(不显著)"))
print(f"AUC 置换检验 p 值: {p_auc:.4f} " + ("(统计学显著)" if p_auc < 0.05 else "(不显著)"))