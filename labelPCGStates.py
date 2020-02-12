import numpy as np
from isempty import isempty

def labelPCGStates(envelope, s1_positions, s2_positions, samplingFrequency, figures):

    envelope = envelope.reshape(-1,1)
    states = np.zeros(len(envelope)).reshape(len(envelope),1)

    ## Timing durations from Schmidt:
    mean_S1 = 0.122*samplingFrequency
    std_S1 = 0.022*samplingFrequency
    mean_S2 = 0.092*samplingFrequency
    std_S2 = 0.022*samplingFrequency

    ## Setting the duration from each R-peak to (R-peak+mean_S1) as the first state:
    # The R-peak in the ECG coincides with the start of the S1 sound (A. G.
    # Tilkian and M. B. Conover, Understanding heart sounds and murmurs: with
    # an introduction to lung sounds, 4th ed. Saunders, 2001.)
    # Therefore, the duration from each R-peak to the mean_S1 sound duration
    # later were labelled as the "true" positions of the S1 sounds:
    for i in range(len(s1_positions)):
        #Set an upper bound, in case the window extends over the length of the
        #signal:
        upper_bound = int(min(len(states), int(s1_positions[i]) + 1 + mean_S1)) # +1 por causa do MATLAB

        #Set the states between the start of the R peak and the upper bound as
        #state 1:
        states[max(1, int(s1_positions[i])):min(upper_bound, len(states))] = 1

    ## Set S2 as state 3 depending on position of end T-wave peak in ECG:
    # The second heart sound occurs at approximately the same time as the
    # end-T-wave (A. G. Tilkian and M. B. Conover, Understanding heart sounds
    # and murmurs: with an introduction to lung sounds, 4th ed. Saunders, 2001.)
    # Therefore, for each end-T-wave, find the peak in the envelope around the
    # end-T-wave, setting a window centered on this peak as the second heart
    # sound state:    

    for i in range(len(s2_positions)):

        #find search window of envelope:
        #T-end +- mean+1sd
        #Set upper and lower bounds, to avoid errors of searching outside size
        #of the signal
        lower_bound = int(max(int(s2_positions[i]) - np.floor(mean_S2 + std_S2), 1))
        upper_bound = int(min(len(states), np.ceil(int(s2_positions[i]) + np.floor(mean_S2 + std_S2)))) + 1 # +1 por causa do MATLAB
        search_window = envelope[lower_bound:upper_bound]*1*(states[lower_bound:upper_bound]!=1)

        # Find the maximum value of the envelope in the search window:
        S2_index = np.argwhere(search_window==max(search_window))[0][0]

        #Find the actual index in the envelope of the maximum peak:
        #Make sure this has a max value of the length of the signal:
        S2_index = min(len(states), lower_bound + S2_index)

        #Set the states to state 3, centered on the S2 peak, +- 1/2 of the
        #expected S2 sound duration. Again, making sure it does not try to set a
        #value outside of the length of the signal:
        upper_bound = int(min(len(states), np.ceil(S2_index + mean_S2/2))) + 1
        states[int(max(np.ceil(S2_index - mean_S2/2),1)):upper_bound] = 3

        #Set the spaces between state 3 and the next R peak as state 4:
        if(i<=len(s2_positions)):
            #We need to find the next R peak after this S2 sound
            #So, subtract the position of this S2 from the S1 positions
            diffs = (s1_positions - float(s2_positions[i]))
            #Exclude those that are negative (meaning before this S2 occured)
            #by setting them to infinity. They are then excluded when finding
            #the minumum later
            diffs[diffs<0] = np.inf

        #If the array is empty, then no S1s after this S2, so set to end of
        #signal:

        if (isempty(diffs<np.inf)):
            end_pos = len(states)
        else:
            #else, send the end position to the minimum diff -1
            index = np.argwhere(diffs==min(diffs))[0][0]
            end_pos = s1_positions[index][0]
        states[int(np.ceil(S2_index + ((mean_S2 + (0*std_S2))/2))):end_pos] = 4

    ## Setting the first and last sections of the signal
    # As all states are derived from either R-peak or end-T-wave locations, the first affirmed state
    # in the signal will always be state 1 or state 3. Therefore, until this state, the
    # first state should always be set to 4 or 2:

    #Find the first step up:
    first_location_of_definite_state = np.argwhere(states!=0)[0][0]

    if(first_location_of_definite_state > 1):

        if(states[first_location_of_definite_state] == 1):
            states[:first_location_of_definite_state] = 4

        if(states[first_location_of_definite_state] == 3):
            states[:first_location_of_definite_state] = 2

    # Find the last step down:
    last_location_of_definite_state = np.argwhere(states!=0)[-1][0]

    if(last_location_of_definite_state > 1):

        if(states[last_location_of_definite_state] == 1):
            states[last_location_of_definite_state:] = 2

        if(states[last_location_of_definite_state] == 3):
            states[last_location_of_definite_state:] = 4

    states = states[:len(envelope)]


    # #Set everywhere else as state 2:
    states[states == 0] = 2
    
    states = states - 1 # -1 para corrigir para índices do Python
    states = states.ravel() # ravel() para compatibilidade com o loop 
                            # for state_i in range(numberOfStates):

    return states