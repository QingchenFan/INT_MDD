import glob
import pandas as pd

# # --------BrainPro HC features --------
sublist = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_allMDD/INTvalue_MDD_Result.csv')
allbehavior = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_allMDD/behaviors/allDataMDD_HAMD.csv')
finalbehavior = pd.merge(allbehavior, sublist, on='subID', how='inner')
finalbehavior.to_csv('./allDataMDD_HAMD_final.csv', index=False)

