class SpringerOptions:
    
    def __init__(self, audio_Fs, audio_segmentation_Fs, segmentation_tolerance, use_mex, include_wavelet_feature):
        self.audio_Fs = audio_Fs
        self.audio_segmentation_Fs = audio_segmentation_Fs
        self.segmentation_tolerance = segmentation_tolerance
        self.use_mex = use_mex
        self.include_wavelet_feature = include_wavelet_feature

def default_Springer_HSMM_options(SpringerOptions):
    springer_options = SpringerOptions
    # The sampling frequency at which to extract signal features:
    springer_options.audio_Fs = 1000
    # The downsampled frequency
    # Set to 50 in Springer paper
    springer_options.audio_segmentation_Fs = 50
    # Set to 50 in Springer paper
    springer_options.segmentation_tolerance = 0.1 # seconds
    # Whether to use the mex code or not:
    springer_options.use_mex = True
    # Whether to use the wavelet function or not:
    springer_options.include_wavelet_feature = True
    return springer_options