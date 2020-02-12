from scipy import signal

def Hilbert_Envelope(input_signal, sampling_frequency):
    hilbert_envelope = abs(signal.hilbert(input_signal))
    return hilbert_envelope