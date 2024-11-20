import glob
import pandas as pd

# # --------HAMD
sublist = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_allMDD/Results/INTvalue_MDD.csv')
allbehavior = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_allMDD/behaviors/BP_Clinical.csv')
finalbehavior = pd.merge(allbehavior, sublist, on='subID', how='inner')
finalbehavior.to_csv('./BP_INTClinical_final.csv', index=False)

