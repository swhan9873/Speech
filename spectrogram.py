# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 20:17:29 2019

@author: gist
"""

import soundfile as sf   
import numpy as np  
import scipy.fftpack as fftpack 
import urllib.request as request
import matplotlib.pyplot as plt  


"""
    To create spectrogram, we reqired 3 locigally distinct function 
                    1. Enframe the audio
                    2. computing the STFT
                    3. computing the power level
"""


x_data, sample_rate = sf.read('ref.wav') # fs : sample rate = sampling frequency
sample_size = len(x_data)
x_timeaxis  = np.arange(0,sample_size/sample_rate,1/sample_rate)

fft_size = 1024

f1=plt.figure(1,figsize=(14,4))
plt.subplot(211)
plt.plot(x_timeaxis,x_data)

"""
Enframe the audio
describe the parameter

x_t[n] = w[n]x[n-tR]

1. x[n] is input data
2. w[n] is window
3. L is window length i.e 0 <= m <= L-1 and w[n] = w[L -1 -n ] 
"""

# enframe the signal
def enframe(x,R,L):
    w = np.hamming(L)
    w1 = 1 # rectangular window.
    w2 = np.bartlett(L)
    frames = []
    nframes = 1 + int((len(x) -L)/R)
    print(len(x))
    print('nframes:',nframes);
    
    for t in range(0,nframes):
        segment = np.copy(x[(t*R):(t*R + L)] * w)
        frames.append(segment)
        
    return frames

w = np.hanning(200)
f2 = plt.figure(figsize=(14,4))
plt.title('Hanning window')
plt.plot(w)

plt.subplot(211)
original_frame = x_data[0:200]
plt.title('the original input data')
plt.plot(np.linspace(0,0.05,len(original_frame)),original_frame)

# R <= L <= N
x_frames = enframe(x_data,R=80,L=200)
plt.figure(figsize=(14,4))
plt.subplot(212)
sub_frame = x_frames[0]
plt.title('the windowig input data')
plt.plot(np.linspace(0,0.05,len(sub_frame)),sub_frame)




# create STFT from the frames

def stft(frames,N,fs):
    
    # how to/?
    stft_frames = [fftpack.fft(x,N) for x in frames]
    freq_axis = np.linspace(0,fs,N)
    return stft_frames,freq_axis

x_stft,x_freqaxis = stft(x_frames,fft_size,sample_rate)
plt.figure(figsize=(14,4))
plt.subplot(211)
plt.plot(x_freqaxis,np.log(np.maximum(1,abs(x_stft[13])**2)))
plt.ylabel('Magnitue squared STFT')
plt.xlabel('Frequency(Hz)')
plt.title('Spectrum of 13 th frame data')


# only care about the frequencise below about 5000Hz
plt.figure(figsize=(14,4))
plt.subplot(211)
plt.plot(x_freqaxis[x_freqaxis<=5000],np.log(abs(x_stft[13][x_freqaxis<=5000])))
plt.ylabel('Magnitue STFT')
plt.xlabel('Frequency(Hz) below about 5000 Hz')

# compute the Power
def stft2level(stft_spectra,max_freq_bin):
    magnitude_spectra = [abs(x) for x in stft_spectra ]
    max_magnitude = max([max(x) for x in magnitude_spectra])
    min_magnitude = max_magnitude / 1000.0
    for t in range(0, len(magnitude_spectra)):
        for k in range(0,len(magnitude_spectra[t])):
            magnitude_spectra[t][k] /= min_magnitude
            if magnitude_spectra[t][k] < 1:
                magnitude_spectra[t][k] =1
            
    level_spectra = [20 * np.log10(x[0:max_freq_bin]) for x in magnitude_spectra]
    return level_spectra


max_freq = 4000

#x_sgram = stft2level(x_stft,int(fft_size*max_freq/sample_rate))
#plt.figure(figsize=(14,4))
#plt.imshow(np.transpose(np.array(x_sgram)),origin='lower',aspect='auto')
#plt.xlabel('Time (ms)')
#plt.ylabel('Freq (Hz)')

    
    
def spectrogram(x,sampling_interval,window_length,N,fs, max_freq):
    
    frames = enframe(x,sampling_interval,window_length)
    spectra, freq_axis = stft(frames,N,fs)
    sgram = stft2level(spectra,int(max_freq*fft_size/fs))
    max_time = len(frames)*sampling_interval/fs
    
    
    return sgram,max_time,max_freq


# 윈도우 길이 L 이 길수록 시간 해상도 감소 하지만 주파수 해상도 는 증가.
# test
R = 128;L =512
x1_sgram,x1_maxtime,x1_maxfreq = spectrogram(
        x                 = x_data,
        sampling_interval = R,
        window_length     = L,
        N                 = fft_size,
        fs                = sample_rate,
        max_freq          = max_freq
)
plt.figure(figsize=(14,4))
plt.imshow(np.transpose(np.array(x1_sgram)),origin='lower',extent=(0,x1_maxtime,0,x1_maxfreq),aspect='auto')
plt.title('Narrowband Spectrogram ')
plt.xlabel('Time (ms)')
plt.ylabel('Freq (Hz)')

# Wideband Spectrogram
x2_sgram,x2_maxtime,x2_maxfreq = spectrogram(
        x                 = x_data,
        sampling_interval = int(0.001*sample_rate),
        window_length     = int(0.004*sample_rate),
        N                 = fft_size,
        fs                = sample_rate,
        max_freq          = max_freq
)
plt.figure(figsize=(14,4))
plt.imshow(np.transpose(np.array(x2_sgram)),origin='lower',extent=(0,x2_maxtime,0,x2_maxfreq),aspect='auto')
plt.title('Wideband Spectrogram ')
plt.xlabel('Time (ms)')
plt.ylabel('Freq (Hz)')    

    
    
    
    
    


