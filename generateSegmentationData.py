def statename(i):
    switcher={
            1:'S1',
            2:'systole',
            3:'S2',
            4:'diastole',
             }
    return switcher.get(i,"invalid state")

def get_data(assigned_states):

    indexList = [0]
    stateList = [statename(assigned_states[0])]

    for i in range(1,len(assigned_states)):
        if (assigned_states[i] != assigned_states[i-1]):
            indexList.append(2*i)
            stateList.append(statename(assigned_states[i]))

    seg_data = dict(index=indexList,state_name=stateList)

    return seg_data