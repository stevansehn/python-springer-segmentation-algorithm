import numpy as np
import scipy.io as scio
import os
import librosa
from default_Springer_HSMM_options import *
from trainSpringerSegmentationAlgorithm import *
from runSpringerSegmentationAlgorithm import *
from plot_figs import *
import generateSegmentationData as gsd
import pathlib
import tqdm

## Load the default options:
# These options control options such as the original sampling frequency of
# the data, the sampling frequency for the derived features and whether the
# mex code should be used for the Viterbi decoding:

springer_options = default_Springer_HSMM_options(SpringerOptions)

## Load the audio data and the annotations:
# These are 6 example PCG recordings, downsampled to 1000 Hz, with
# annotations of the R-peak and end-T-wave positions.
mat = scio.loadmat('example_data.mat')
example_data = mat['example_data']

## Split the data into train and test sets:
# Select the 5 recordings for training and a sixth for testing:
training_indices = [0, 46, 360, 401, 571] # MATLAB 1, 47, 361, 402, 572
train_recordings = []
train_annotations = []
for i in training_indices:
    train_recordings.append(example_data[0,0]['example_audio_data'][0][i])
    train_annotations.append(example_data[0][0]['example_annotations'][i][:])

## Train the HMM:
B_matrix, pi_vector, total_obs_distribution = trainSpringerSegmentationAlgorithm(train_recordings,train_annotations,springer_options.audio_Fs)

inputRoot = '/media/linse/dados/stevan/datasets/heart-sound/training'
inputRoot = pathlib.Path(inputRoot)

outputRoot = '/media/linse/dados/stevan/datasets/heart-sound/python_segmentation/'
outputRoot = pathlib.Path(outputRoot)
outputRoot.mkdir(parents=True, exist_ok=True)

# Coloca a lista em ordem alfabética
folderList = sorted(list(inputRoot.glob('training*')))

# 'training-a/', 'training-b/', 'training-c/', 'training-d/', 'training-e/', 'training-f/'
for i in tqdm.trange(len(folderList)):
    fname = folderList[i].name
    
    inFolder = inputRoot / fname
    outFolder = outputRoot / fname
    outFolder.mkdir(parents=True, exist_ok=True)

    fileList = sorted(list(inFolder.glob('*.wav')))
    
    # fileList = fileList[:2]

    for k in tqdm.trange(len(fileList)):
        recordName = inFolder / fileList[k]
        outFile = (outFolder / fileList[k].stem).with_suffix('.npz')
        
        # print(f'{folderList[i].name} | File {k+1} of {len(fileList)} | {fileList[k].stem}')
        
        # Load data and resample data
        PCG_resampled = librosa.load(recordName,sr=springer_options.audio_Fs)[0]
        
        assigned_states = runSpringerSegmentationAlgorithm(PCG_resampled, springer_options.audio_Fs, B_matrix, pi_vector, total_obs_distribution, False)
        seg_data = gsd.getdata(assigned_states)
        np.savez(outFile, **seg_data)