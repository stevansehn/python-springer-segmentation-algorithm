import numpy as np
from scipy.signal.spectral import spectrogram

def get_PSD_feature_Springer_HMM(data, sampling_frequency, frequency_limit_low, frequency_limit_high):

    WINDOW = int(sampling_frequency/40) # 25
    NOVERLAP = 13 # round(sampling_frequency/80) # 13 no MATLAB e 12 no Python
    NFFT = sampling_frequency

    # Find the spectrogram of the signal:
    [F, T, P] = spectrogram(data, fs=sampling_frequency, window='hanning', nperseg=WINDOW, 
                                noverlap=NOVERLAP, nfft=NFFT, mode='psd')

    minimum = min(abs(F - frequency_limit_low))
    l = abs(F - frequency_limit_low)
    low_limit_position = int(np.argwhere(l==minimum))
    maximum = min(abs(F - frequency_limit_high))
    h = abs(F - frequency_limit_high)
    high_limit_position = int(np.argwhere(h==maximum))

    # Find the mean PSD over the frequency range of interest:
    psd = np.mean(P[low_limit_position:high_limit_position][:],axis=0).ravel()
    
    return psd