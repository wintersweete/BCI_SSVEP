from __future__ import print_function
"""
Created on Wed Jan 02 09:03:53 2019

@author: meixuer
"""
"""
This code is for SSVEP.
1. there is not bandtop filter of 50Hz.
2. you can choose 1th/2th harmonic or 1th/2th/3th harmonic hy setting 'harmonic = 2' or 'harmonic = 3'.
3. the streams from outside must only be marked at the start of trial and the end as 'start' and 'end' 
4. when you use this code you must ensure the marker coding time is less than time of ssvep
5. the time of sending results of decoding must at the last of the time of inlet in MATLAB
6. you must run MATLAB when python has ran and show 'Please start MATLAB'
"""
import numpy as np
import scipy.signal as sig
import os

from mne.decoding import CSP
from sklearn.cross_decomposition import CCA
from mne.time_frequency import psd_array_welch

from pylsl import StreamInlet, resolve_stream
from pylsl import StreamInfo, StreamOutlet
import time
import random



# #### setting parameter for ssvep ######
channels = [30, 31, 32] # the channel used for ssvep
harmonic = 2            # the number of harmonic for generating reference
low = 4                 # the frequency of butter-worth filtering
high = 20
n_components = 1        # the components for CCA during computing
sampling_rate = 1024    # deciding by the sampling rate you send to EEG-hat
epoch_length = 1.5      # units:  mrkss; the length of windows
epoch_step = 0.2        # units: s; the overlap of windows
Fre = [8,9,10,11,12,13,14,15]        # units: Hz; deciding by experimental paradigm
deciding_num = 3        # the same results of windows for deciding the output
time_ssvep = 5          # the time of ssvep for person

########## information save ###########
print('Please enter the /date_name/(2019.3.21_Lisi) of participant:')
cur_dir = '/home/qin/Desktop/SSVEP/data'
folder_name = raw_input()
if os.path.isdir(cur_dir):
    os.mkdir(os.path.join(cur_dir, folder_name))
f = open(cur_dir+'/'+folder_name+'/information.txt', 'a')
f.write('Filter:'+str(low)+'-'+str(high)+'\n')
f.write('Sampling_rate:'+str(sampling_rate)+'\n')
f.write('Frequency for ssvep:' + str(Fre)+'\n')
f.write('The length of data (unit: s) for each trial:' + str(time_ssvep))
f.write('The channels of EEG data:' + str(len(channels)))
f.close()


# #### flitering #####
def filtering(epoch, sampling_rate, low, high):
    # epoch is a sample, the size is the sampling_point * channels
    for i in range(np.size(epoch, 0)):
        signal = epoch[:, i]
        lowcut = low/(sampling_rate*0.5)
        highcut = high/(sampling_rate*0.5)
        [b, a] = sig.butter(4, [lowcut, highcut], 'bandpass')
        epoch[:, i] = sig.filtfilt(b, a, signal)
        return epoch

# #### map-min-max #####
def minmax(epoch):
    for i in range(len(epoch)):
        epoch[i] = (epoch[i]-np.min(epoch[i]))/float(np.max(epoch[i])- np.min(epoch[i]))
    return np.transpose(epoch)



# #### generation of data  #####
def signalReference(T, fre, harmonic):
    reference = np.zeros(shape=(len(T), 2*harmonic))
    reference[:, 0] = np.array([np.sin(2*np.pi*i*fre) for i in T])  # sin
    reference[:, 1] = np.array([np.cos(2*np.pi*i*fre) for i in T])  # cos
    reference[:, 2] = np.array([np.sin(2*np.pi*i*fre*2) for i in T])  # 2sin
    reference[:, 3] = np.array([np.cos(2*np.pi*i*fre*2) for i in T])  #2cos
    if (harmonic == 3):
        reference[:, 4] = np.array([np.sin(2*np.pi*i*fre*3) for i in T])  # 3sin
        reference[:, 5] = np.array([np.cos(2*np.pi*i*fre*3) for i in T])  # 3cos
    return reference

# #### compution value of CSP #####
'''
    psds, freqs = psd_array_welch(epoch, F, fmin=low, fmax=high, n_fft=256, n_overlap=0, n_per_seg=None, n_jobs=1, verbose=None)
    psd_mean = np.sum(psds, axis=2)
    psd_mean = np.reshape(psd_mean, (psd_mean.shape[0], psd_mean.shape[1], 1))
    if fmin == 5:
        Pre_data = psd_mean
    else:
        Pre_data = np.concatenate((Pre_data, psd_mean), axis=2)
'''
# #### compution value of CCA #####
def correlate(epoch, n_components, reference):
    cca = CCA(n_components = n_components)
    cca.fit(epoch, reference)
    u, v = cca.transform(epoch, reference)
    corr = np.corrcoef(u.T, v.T)[0, 1]
    return corr



# main()
Real = []
Pre = []
# generating the reference
T = np.linspace(0, epoch_length, num = int(sampling_rate*epoch_length))
reference = []
for i in Fre:
    reference.append(signalReference(T, i, harmonic))

# the send results streams
#info = StreamInfo('ResultMarker', 'Markers', 1, 0, 'string', 'myuniquesourceid24445')
#outlet = StreamOutlet(info)

# first resolve an EEG stream on the lab network
streams_data = resolve_stream('type', 'EEG')
#streams_marker = resolve_stream('name', 'SSVEPMarkerStream')

# create a new inlet to read from the stream
inlet_data = StreamInlet(streams_data[0])
#inlet_marker = StreamInlet(streams_marker[0])



time.sleep(2)
print('Please start MATLAB .....')

trial = 1
while True:
    start_time = 0.0
    corr = [0]*len(Fre)
    Corr = [0]*len(Fre)
    OUT = 0
    #markers, timestamp_marker = inlet_marker.pull_chunk()
    data, timestamp_data = inlet_data.pull_chunk()  # pull_chunk
    data = np.array(data)
    Epoch = np.zeros([1, len(channels)])
    if data.shape[0] > 0:
        Epoch = data[:, channels]

    # wo need read a epoch_length(1.5)+0.5 firstly noting: pull_chunk just for 1s
    time.sleep(1)
    data, timestamp_data = inlet_data.pull_chunk()
    #markers, timestamp_marker = inlet_marker.pull_chunk()
    data = np.array(data)
    data = data[:, channels]
    Epoch = np.vstack((Epoch, data))
    time.sleep(1)

    while (Epoch.shape[0] < int(sampling_rate*(time_ssvep))):

        data, timestamp_data = inlet_data.pull_chunk()
        #markers, timestamp_marker = inlet_marker.pull_chunk()
        data = np.array(data)
        data = data[:, channels]
        Epoch = np.vstack((Epoch, data))
        epoch = Epoch[-int(sampling_rate * epoch_length+1):-1, :]
        filtered_data = filtering(epoch, sampling_rate, low, high)
        #epoch = minmax(np.transpose(filtered_data))
        corr = []
        for ref in reference:
            corr.append(correlate(epoch, n_components, ref))
        #corr[0] = correlate(epoch, n_components, reference1)
        #corr[1] = correlate(epoch, n_components, reference2)
        #corr[2] = correlate(epoch, n_components, reference3)
        np.set_printoptions(precision=3)
        print(Fre[corr.index(max(corr))], "\t", np.array(corr))
        if max(Corr) == deciding_num:
            pass
        else:
            Corr[corr.index(max(corr))] = Corr[corr.index(max(corr))]+1
        time.sleep(epoch_step)
    # saving EEG data of each trial
    f = open(cur_dir + '/' + folder_name + '/EEGdata.txt', 'a')
    f.write(str(trial) + ':' + str(Epoch[0:int(sampling_rate * time_ssvep) - 1, :]) + '\n')
    f.close()
    trial = trial + 1

    print(Corr)
    #OUT = Corr.index(max(Corr))
    '''
    if 0 == OUT:
        outlet.push_sample(['Left'])  # sending
    elif 1 == OUT:
        outlet.push_sample(['Center'])  # sending
    elif 2 == OUT:
        outlet.push_sample(['Right'])  # sending
    '''
    print('sent')
    Pre.append(corr.index(max(corr)))

    time.sleep(0.5)
