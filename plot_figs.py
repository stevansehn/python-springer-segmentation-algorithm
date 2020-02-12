import numpy as np
import matplotlib.pyplot as plt
from normalise_signal import *

def plot_fig(audio_data, Fs, assigned_states, left=0, right=3):
    plt.figure(figsize=(20,10))
    t1 = np.arange(len(audio_data))/Fs
    plt.plot(t1,normalise_signal(audio_data),'k', label='Audio data')
    plt.plot(t1,assigned_states+1,'r--',label='Derived states')
    plt.xlabel('Time (s)')
    plt.xlim(left, right) 
    plt.legend(fancybox=True, shadow=True, prop={'size': 20})
    plt.show()
    
def compare_figs(audio_data, Fs, assigned_states, MATLAB_states, left=0, right=3):
    plt.figure(figsize=(20,10))
    t1 = np.arange(len(audio_data))/Fs
    plt.plot(t1,normalise_signal(audio_data),'k', label='Audio data')
    plt.plot(t1,assigned_states+1,'r--',label='Derived states')
    plt.plot(t1,MATLAB_states,'b--',label='MATLAB')
    plt.xlabel('Time (s)')
    plt.xlim(left, right) 
    plt.legend(fancybox=True, shadow=True, prop={'size': 20})
    plt.show()

def save_figs(audio_data, Fs, assigned_states, states, spath, left=0, right=3):
    plt.figure(figsize=(20,10))
    t1 = np.arange(len(audio_data))/Fs
    plt.plot(t1,normalise_signal(audio_data),'k', label='Audio data')
    plt.plot(t1,assigned_states+1,'r--',label='Derived states')
    plt.plot(t1,states,'b--',label='MATLAB')
    plt.xlabel('Time (s)')
    plt.xlim(left, right) 
    plt.legend(fancybox=True, shadow=True, prop={'size': 20})
    plt.savefig(spath.with_suffix(spath.suffix + '.png'))
    plt.savefig(spath.with_suffix(spath.suffix + '.pdf'))
    plt.close()