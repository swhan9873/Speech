# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:01:39 2019

@author: gist
"""

"""
목적 : wav 파일 하나씩 불러와서 (같은 사람임) d-vector 을 생성해보자. 특히 주의할점은
프레임 별 어떻게 할 것인가? MFCC or log mel filterbank 를 어떻게 생성 할 것인지.. 맨날 맨날
안했는데 이젠 진짜 해보자. 시발. 상욱ㅇ ㅏ정신 차리ㅏ자 ㅇㅋ?

"""
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn


sr = 16000
nfft = 512 # for mel spectrogram preprocess
window = 0.025 #(s) # window*sr = 400
hop = 0.01 #(s)     # hop*sr    = 160
nmels =  40 # number of mel energy
frame = 160 # max number of time steps in input after preprocess

class SpeechEmbedder(nn.Module):
    def __init__(self):
        super(SpeechEmbedder,self).__init__()
        self.lstm = nn.LSTM(
                input_size=nmels,
                hidden_size=768,
                num_layers=3,
                batch_first=True
        )
        self.proj = nn.Linear(768,256)
    
    def forward(self,mel):
        mels = mel.unfold(1,80,40) # Extracts sliding local blocks from an batched input tensor.
        
        print('unfold',mels.shape) # (40, 197) --> (40, 3, 80)
        
        mels = mels.permute(1,2,0) # (40, 3, 80) --> (3, 80. 40)
        print('11',mels)
        print('permutate',mels.shape)
        
        x,_ = self.lstm(mels) #(3, 80, 786)
        print('lstm',x.shape)
        x = x[:,-1,:] # use last frame only
        print('last frame only',x.shape) #(3, 786)
        x = self.proj(x)
        print('lineaer',x.shape)         # (3, 256)
        print(x)
        
        # L2-normalization
        x_L2 = x / torch.norm(x,p=2,dim=1,keepdim=True)
        print(x_L2)        
        
        print('x.size(0):',x_L2.size(0))
        
        dvec = x_L2.sum(0) / x_L2.size(0) # average pooling over time frames.

        return dvec
################################################################################

utter, sr = librosa.core.load('./19_A.flac',sr)

STFT = librosa.core.stft(
        y       = utter,
        n_fft   = nfft,
        win_length = int(window * sr),
        hop_length = int(hop * sr) 
)
print('STFT size:',STFT.shape) # (257,197) 197 은 utterance (31440) / hop_length (160) = 196.5 의 결과

magnitudes = np.abs(STFT)**2

print(magnitudes)


# Create a filterbank matrix to combine FFT bins into Mel-frequency bins
# 40-dimentional filterbank
mel_basis = librosa.filters.mel(sr=sr,n_fft=nfft,n_mels=nmels) # (40,257)

# mel spectrogram.
mel = np.dot(mel_basis,magnitudes)
#librosa.display.specshow(mel,x_axis='time',y_axis='mel',sr=sr,fmax=8000)

# log mel spectrogram.
log_mel = np.log10(mel + 1e-6)

plt.figure(figsize=(9, 5))
librosa.display.specshow(log_mel,x_axis='time',y_axis='mel',sr=sr,fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('log-Mel-frequency spectrogram')
plt.tight_layout()
plt.show()

first_160_frame = log_mel[:, :frame] # index 0:160  (diff: 160)
last_160_frame =  log_mel[:,-frame:] # index 37:197 (197 이 값은 신호에 따라 모두 다름) (diff: 160)


utterance_spec = list()
utterance_spec.append(first_160_frame)
utterance_spec.append(last_160_frame)

utterance_spec =np.array(utterance_spec) # 2, 40, 160


#################################################################################

## mel spectrogram library
#mel2 = librosa.feature.melspectrogram(y=utter,sr=sr,n_mels=40)
#S_db = librosa.power_to_db(mel2,ref=np.max)
#librosa.display.specshow(S_db)

""" Create d-vector """
# change the Tensor.
dvec_mel = torch.from_numpy(log_mel).float()
print('dvec_mel',dvec_mel.shape)

embedder = SpeechEmbedder()
d_vector = embedder(dvec_mel)

print(d_vector.shape)



