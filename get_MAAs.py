import numpy as np

def get_MAAs(sampleframes):
    M = []
    for i in range(len(sampleframes[0,:])):
        M.append(max(abs(sampleframes[:,i])))
    MAAs = np.asarray(M).T
    return MAAs