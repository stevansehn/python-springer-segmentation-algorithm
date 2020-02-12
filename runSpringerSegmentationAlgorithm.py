import matplotlib.pyplot as plt
from default_Springer_HSMM_options import *
from getSpringerPCGFeatures import *
from getHeartRateSchmidt import *
from viterbiDecodePCG_Springer import *
from expand_qt import *

def runSpringerSegmentationAlgorithm(audio_data, model):

    ## Load the default options:

    springer_options = default_Springer_HSMM_options(SpringerOptions)
    Fs = springer_options.audio_Fs

    ## Get model parameters

    B_matrix, pi_vector, total_obs_distribution = model

    ## Get PCG Features:

    PCG_Features, featuresFs = getSpringerPCGFeatures(audio_data, Fs)

    ## Get PCG heart rate

    heartRate, systolicTimeInterval = getHeartRateSchmidt(audio_data, Fs)

    qt, _, _  = viterbiDecodePCG_Springer(PCG_Features, pi_vector, B_matrix, total_obs_distribution, heartRate, systolicTimeInterval, featuresFs)

    assigned_states = expand_qt(qt, featuresFs, Fs, len(audio_data))
    assigned_states[:10].reshape(-1,1)

    return assigned_states