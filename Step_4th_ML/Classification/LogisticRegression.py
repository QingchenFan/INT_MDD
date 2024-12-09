from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd



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
# Initialize the Logistic Regression model
logreg = LogisticRegression()

# Initialize Stratified KFold cross-validator
strat_k_fold = StratifiedKFold(n_splits=10)

# Perform KFold cross-validation
scores = cross_val_score(logreg, x_data, y_label, scoring='accuracy', verbose=6,cv=strat_k_fold)
print(scores)
# Calculate the mean accuracy
mean_accuracy = scores.mean()
print(mean_accuracy)
