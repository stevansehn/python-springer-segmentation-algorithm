import numpy as np
import librosa
from default_Springer_HSMM_options import *
from butterworth_filters import *
from schmidt_spike_removal import *
from normalise_signal import *
from Homomorphic_Envelope_with_Hilbert import *
from Hilbert_Envelope import *
from get_PSD_feature_Springer_HMM import *
from getDWT import *

def getSpringerPCGFeatures(audio_data, Fs, figures=False):
    # function PCG_Features = getSpringerPCGFeatures(audio, Fs)
    # Get the features used in the Springer segmentation algorithm.
    
    springer_options = default_Springer_HSMM_options(SpringerOptions)
    
    ## Check to see if the Wavelet toolbox is available on the machine:
    include_wavelet = springer_options.include_wavelet_feature 
    featuresFs = springer_options.audio_segmentation_Fs # Downsampled feature sampling frequency

    ## 25-400Hz 4th order Butterworth band pass
    audio_data = butterworth_low_pass_filter(audio_data,2,400,Fs)
    audio_data = butterworth_high_pass_filter(audio_data,2,25,Fs)
    
    ## Spike removal from the original paper:
    audio_data = schmidt_spike_removal(audio_data, Fs)
    
    ## Find the homomorphic envelope
    homomorphic_envelope = Homomorphic_Envelope_with_Hilbert(audio_data, Fs)
    # Downsample the envelope:
    downsampled_homomorphic_envelope = librosa.resample(homomorphic_envelope, Fs, featuresFs)
    # normalise the envelope:
    downsampled_homomorphic_envelope = normalise_signal(downsampled_homomorphic_envelope)
    
    ## Hilbert Envelope

    hilbert_envelope = Hilbert_Envelope(audio_data, Fs)
    downsampled_hilbert_envelope = librosa.resample(hilbert_envelope, Fs, featuresFs)
    downsampled_hilbert_envelope = normalise_signal(downsampled_hilbert_envelope)

    ## Power spectral density feature:

    psd = get_PSD_feature_Springer_HMM(audio_data, Fs, 40,60)
    psd_resampled = librosa.resample(psd, len(psd), len(downsampled_homomorphic_envelope))
    psd_resampled = psd_resampled[:downsampled_homomorphic_envelope.size] # garante len certo, por algum motivo, librosa.resample está dando len + 1
    psd_resampled = normalise_signal(psd_resampled).reshape(-1,1)
    psd_resampled = psd_resampled.ravel()

    ## Wavelet features

    if(include_wavelet):
        wavelet_level = 3
        wavelet_name ='rbio3.9'

        # Audio needs to be longer than 1 second for getDWT to work:
        if(len(audio_data)< Fs*1.025):
            audio_data = np.hstack((audio_data,np.zeros(round(0.025*Fs))))

        [cD, cA] = getDWT(audio_data,wavelet_level,wavelet_name);

        wavelet_feature = abs(cD[wavelet_level-1][:])
        wavelet_feature = wavelet_feature[:len(homomorphic_envelope)]
        downsampled_wavelet = librosa.resample(wavelet_feature, Fs, featuresFs)
        downsampled_wavelet =  normalise_signal(downsampled_wavelet).reshape(-1,1)
        downsampled_wavelet = downsampled_wavelet.ravel()
        
    if(include_wavelet):
        PCG_Features = [downsampled_homomorphic_envelope, downsampled_hilbert_envelope, psd_resampled, downsampled_wavelet]
    else:
        PCG_Features = [downsampled_homomorphic_envelope, downsampled_hilbert_envelope, psd_resampled]
    
    PCG_Features = np.array(PCG_Features).T # Para ficar igual ao MATLAB

    return PCG_Features, featuresFs