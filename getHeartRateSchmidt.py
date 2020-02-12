import numpy as np
from butterworth_filters import *
from schmidt_spike_removal import *
from Homomorphic_Envelope_with_Hilbert import *


def getHeartRateSchmidt(audio_data, Fs):

    audio_data = audio_data.ravel()
    
    ## Get heatrate:
    # From Schmidt:
    # "The duration of the heart cycle is estimated as the time from lag zero
    # to the highest peaks between 500 and 2000 ms in the resulting
    # autocorrelation"
    # This is performed after filtering and spike removal:

    ## 25-400Hz 4th order Butterworth band pass
    audio_data = butterworth_low_pass_filter(audio_data,2,400,Fs)
    audio_data = butterworth_high_pass_filter(audio_data,2,25,Fs)

    ## Spike removal from the original paper:
    audio_data = schmidt_spike_removal(audio_data, Fs)

    ## Find the homomorphic envelope
    homomorphic_envelope = Homomorphic_Envelope_with_Hilbert(audio_data, Fs)

    ## Find the autocorrelation:
    y = homomorphic_envelope - np.mean(homomorphic_envelope)
    c = np.correlate(y,y,'full')
    signal_autocorrelation = c[len(homomorphic_envelope):-1]

    min_index = int(0.5*Fs)
    max_index = int(2*Fs)

    if max_index > len(signal_autocorrelation):
        max_index = len(signal_autocorrelation)

    index = np.argwhere(signal_autocorrelation == max(signal_autocorrelation[min_index:max_index]))[0][0]
    # true_index = index + min_index - 1
    true_index = index

    heartRate = 60/true_index*Fs

    ## Find the systolic time interval:
    # From Schmidt: "The systolic duration is defined as the time from lag zero
    # to the highest peak in the interval between 200 ms and half of the heart
    # cycle duration"

    max_sys_duration = int(round(((60/heartRate)*Fs)/2))
    min_sys_duration = int(round(0.2*Fs))

    pos = np.argwhere(signal_autocorrelation == max(signal_autocorrelation[min_sys_duration:max_sys_duration]))[0][0]
    # systolicTimeInterval = (min_sys_duration+pos)/Fs
    systolicTimeInterval = (pos)/Fs

    return heartRate, systolicTimeInterval