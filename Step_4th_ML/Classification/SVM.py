
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import cohen_kappa_score
import numpy as np
import pandas as pd
from sklearn import svm


Data = pd.read_csv("/Volumes/QC/INT/INT_BN246_HC135BP_allMDD/Results/INTvalue_HCMDD.csv")

region = pd.read_csv("/Volumes/QC/INT/INT_BN246_HC135BP_allMDD/Results/signifRegion.csv")

regions = region.columns.tolist()

MDDData = Data[Data['disorder'] == 1].sample(n=222)
HCData = Data[Data['disorder'] == 0]

Data = pd.concat([HCData, MDDData])
# all Regions feature
brainRegion = Data.columns.tolist()
del brainRegion[:2]

x_data = np.array(Data[regions])

y_label = np.array(Data['disorder'])

kf = KFold(n_splits=10,shuffle=True)
acc_res = []
kappa_res = []
for train_index, test_index in kf.split(x_data):

    # split data
    X_train, X_test = x_data[train_index, :], x_data[test_index, :]

    y_train, y_test = y_label[train_index], y_label[test_index]


    # Model
    svmmodel = svm.SVC(kernel='linear')

    svmmodel.fit(X_train, y_train)
    t_score = svmmodel.score(X_train, y_train)
    #print('t_score', t_score)
    Predict_Score = svmmodel.predict(X_test)
    #print('-Predict_Score-', Predict_Score)
    print('y_test:',y_test)
    print('Predict_Score:',Predict_Score)

    acc = accuracy_score(y_test, Predict_Score)
    print('-acc = %.2f:' %(acc))
    acc_res.append(float("%.2f"%(acc)))

    kappa = cohen_kappa_score(np.array(y_test).reshape(-1, 1), np.array(Predict_Score).reshape(-1, 1))
    print('-kappa = %.2f:' %(kappa))
    kappa_res.append(kappa)


    # 通过测试集的预测结果
    # 打印出三种评估指标的分类报告进行模型评估
    #print(metrics.classification_report(y_test, Predict_Score))

print(acc_res)
print('Result: acc = %.3f, kappa = %.3f ' % (np.mean(acc_res), np.mean(kappa_res)))