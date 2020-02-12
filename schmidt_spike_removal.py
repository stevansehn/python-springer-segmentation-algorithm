import numpy as np
from statistics import median

def get_MAAs(sampleframes):
    M = []
    for i in range(len(sampleframes[0,:])):
        M.append(max(abs(sampleframes[:,i])))
    MAAs = np.asarray(M).T
    return MAAs

def schmidt_spike_removal(original_signal, fs):
    
    # Find the window size (500 ms)
    windowsize = round(fs/2)

    # Find any samples outside of a integer number of windows:
    trailingsamples = len(original_signal) % windowsize

    # Reshape the signal into a number of windows:
    sampleframes = original_signal[:len(original_signal)-trailingsamples]
    sampleframes = sampleframes.reshape(windowsize,-1,order='F')

    # Find the MAAs:
    MAAs = get_MAAs(sampleframes)

    # While there are still samples greater than 3* the median value of the
    # MAAs, then remove those spikes:
    emptyarray = np.nonzero(MAAs>median(MAAs)*3)[0]

    while(not emptyarray.size==0):

        # Find the window with the max MAA:
        val = max(MAAs)
        window_num = np.argmax(MAAs)

        if(window_num.size>1):
            window_num = window_num[0]

        # Find the postion of the spike within that window:
        val = max(abs(sampleframes[:,window_num]))
        spike_position = np.argmax(abs(sampleframes[:,window_num]))

        if(spike_position.size>1):
            spike_position = spike_position[0]

        # Finding zero crossings (where there may not be actual 0 values, just a change from positive to negative):
        zero_crossings = np.hstack((abs(np.diff(np.sign(sampleframes[:,window_num])))>1, 0))

        # Find the start of the spike, finding the last zero crossing before
        # spike position. If that is empty, take the start of the window:
        if (not np.nonzero(zero_crossings[:spike_position+1])[0].size):
            spike_start = 0
        else:
            nonzero = np.nonzero(zero_crossings[:spike_position+1])[0][-1] #[-1] = MATLAB "last"
            spike_start = np.max((0,nonzero))

        # Find the end of the spike, finding the first zero crossing after
        # spike position. If that is empty, take the end of the window:
        zero_crossings[:spike_position+1] = 0
        if (not np.nonzero(zero_crossings)[0].size):
            spike_end = windowsize
        else:
            nonzero = np.nonzero(zero_crossings)[0][0] #[0] = MATLAB "first"
            spike_end = np.min((nonzero, windowsize))

        # Set to Zero
        sampleframes[spike_start:spike_end,window_num] = 0.0001

        # Recaclulate MAAs
        MAAs = get_MAAs(sampleframes)

        emptyarray = np.nonzero(MAAs>median(MAAs)*3)[0]

    despiked_signal = sampleframes.reshape(sampleframes.size,order='F')

    # Add the trailing samples back to the signal:
    despiked_signal = np.concatenate((despiked_signal, original_signal[despiked_signal.size:]))

    return despiked_signal