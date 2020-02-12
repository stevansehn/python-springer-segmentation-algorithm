import numpy as np
from trainBandPiMatricesSpringer import *
from getSpringerPCGFeatures import *
from labelPCGStates import *


def trainSpringerSegmentationAlgorithm(PCGCellArray, annotationsArray, Fs):

    numberOfStates = 4 
    numPCGs = len(PCGCellArray) 

    # A matrix of the values from each state in each of the PCG recordings:
    state_observation_values = np.empty((numPCGs, numberOfStates))
    state_observation_values[:] = np.nan
    state_observation_values = state_observation_values.tolist()

    for PCGi in range(numPCGs):

        PCG_audio = PCGCellArray[PCGi].ravel()

        S1_locations = annotationsArray[PCGi][0] - 1
        S2_locations = annotationsArray[PCGi][1] - 1

        PCG_Features, featuresFs = getSpringerPCGFeatures(PCG_audio, Fs)

        PCG_states = labelPCGStates(PCG_Features[:,0], S1_locations, S2_locations, featuresFs, False)

        ## Group together all observations from the same state in the PCG recordings:
        for state_i in range(numberOfStates):
            state_observation_values[PCGi][state_i] = PCG_Features[:][PCG_states==state_i]

    # Save the state observation values to the main workspace of Matlab for
    # later investigation if needed:
    # assignin('base', 'state_observation_values', state_observation_values) # Específico do MATLAB; cria uma variável global, mas no Python não sei se isso é possível dado que esta função será um .py separado

    # Train the B and pi matrices after all the PCG recordings have been labelled:
    logistic_regression_B_matrix, pi_vector, total_obs_distribution = trainBandPiMatricesSpringer(state_observation_values)

    return logistic_regression_B_matrix, pi_vector, total_obs_distribution