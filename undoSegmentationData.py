import numpy as np

def statenum(i):
    switcher={
            'S1':1,
            'systole':2,
            'S2':3,
            'diastole':4,
             }
    return switcher.get(i,"invalid state")

def get_states(MATLAB_seg_data, assigned_states_size, key):
    indexList = np.floor(MATLAB_seg_data[astr][:,0]/2)
    stateList = MATLAB_seg_data[astr][:,1]

    MATLABstates = []
    j = 1
    for i in range(assigned_states_size):
        MATLABstates.append(statenum(stateList[j-1]))
        if (i == indexList[j]):
            if (j<indexList.size-1):
                j += 1
            else:
                j = indexList.size-1

    return np.array(MATLABstates)