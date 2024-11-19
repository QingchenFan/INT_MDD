import glob

import numpy as np


path =  '/Volumes/QCI/NormativeModel/BrainProject/BPMDD_246timeseries/*/*.txt'
databox = glob.glob(path)

for i in databox:
    print(i)
    subID = i.split('/')[-2]

    a = np.loadtxt(i).shape[0]

    res = np.full((a,1),0.1)

    #outp = '/Volumes/QCI/NormativeModel/DuiLie/MDD/DLMDDData/'+subID+'/'+subID+'_FD.txt'
    outp =  '/Volumes/QCI/NormativeModel/BrainProject/BPMDD_246timeseries/'+subID+'/'+subID+'_FD.txt'
    np.savetxt(outp, res, fmt='%.1f', delimiter=',')
