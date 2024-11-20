import glob
import nibabel as nib
import numpy as np
import os

path = '/Volumes/QCI/NormativeModel/Data135/HC/dtseriesnii/sub-HC046*ap*'
databox = glob.glob(path)

for i in databox:
    subID = i.split('/')[-1][0:9]
    print(subID)

    newpath = '/Volumes/QCI/NormativeModel/Data135/HC/Data135HCData/' + subID
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    data = nib.load(i).get_fdata()

    for j in range(90):
        start_index = j * 1000
        end_index = (j + 1) * 1000
        print(j,'|','start:',start_index,'--end:',end_index)

        extracted_matrix = data[:, start_index:end_index]
        print('extracted_matrix-',extracted_matrix.shape)

        newdatap = newpath + '/' + subID +'_'+ str(j) +'.txt'
        #np.savetxt(newdatap, extracted_matrix)
    lastmatrix = data[:,90000:]
    print('lastmatrix.shape',lastmatrix.shape)
    lp = newpath + '/' + subID +'_90' +'.txt'
    #np.savetxt(lp, lastmatrix)


