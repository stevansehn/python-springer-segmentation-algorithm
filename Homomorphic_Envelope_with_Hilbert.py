import numpy as np
from scipy import signal

def Homomorphic_Envelope_with_Hilbert(input_signal, sampling_frequency, lpf_frequency=8):
    
    # 8Hz, 1st order, Butterworth LPF
    b, a = signal.butter(1, 2*lpf_frequency/sampling_frequency, 'low')
    input_signal = np.log(abs(signal.hilbert(input_signal)))
    low_pass_filtered_signal = signal.filtfilt(b, a, input_signal, padtype='odd')
    homomorphic_envelope = np.exp(low_pass_filtered_signal)
    
    # Remove spurious spikes in first sample:
    homomorphic_envelope[0] = homomorphic_envelope[1]
    
    return homomorphic_envelope