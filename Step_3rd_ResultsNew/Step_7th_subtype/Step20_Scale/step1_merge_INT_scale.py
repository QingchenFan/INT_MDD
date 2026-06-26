import numpy as np
import pandas as pd

file_1 ='/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_NetworkHierarchy/INT_DiffNetwork/' \
        'subtype2_DiffNetwork_mean.csv'
file_2 ='/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step19_Scale/subtype2_GAD.csv'
#file_1 = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype2_FirstEp.csv'
#Read the CSV files into DataFrames
df1 = pd.read_csv(file_1)
df2 = pd.read_csv(file_2)

df_new = pd.merge(df2, df1, on='subID', how='inner')
df_new.to_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step1_NetworkHierarchy/INT_DiffNetwork/DdiffNetwork_scale_Corre/'
              'subtype2_DiffNetwork_GAD.csv', index=False, encoding='utf-8-sig')