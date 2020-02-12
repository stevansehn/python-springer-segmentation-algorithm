from scipy import signal
import numpy as np

def butterworth_low_pass_filter(original_signal,order,cutoff,sampling_frequency, figures=False):
    Wn = 2*cutoff/sampling_frequency
    b, a = signal.butter(order, Wn, 'low', analog=False)
    low_pass_filtered_signal = signal.filtfilt(b, a, original_signal.ravel(), padtype='odd')
    
    return low_pass_filtered_signal

def butterworth_high_pass_filter(original_signal,order,cutoff,sampling_frequency, figures=False):
    Wn = 2*cutoff/sampling_frequency
    b, a = signal.butter(order, Wn, 'high', analog=False)
    high_pass_filtered_signal = signal.filtfilt(b, a, original_signal.ravel(), padtype='odd')
    
    return high_pass_filtered_signal