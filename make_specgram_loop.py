import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import math
from scipy import hamming
from scipy import signal

fs = 100
lenWindow = 512
over_lap = 510
hammWindow = hamming(lenWindow)

name = ['ACC_X','ACC_Y','ACC_Z','GYRO_X','GYRO_Y','GYRO_Z','EOG_L','EOG_R','EOG_H','EOG_V']

path = ['180117_G1_JINSMEME', '180117_H1_JINSMEME', '180112_E1_JINSMEME','171227_C1_JINS_MEME']


"""
G1 0:65527 1:27381
H1 0:21907 1:37107
E1 0:27336 1:23051
C1 0:29526 1:89126
"""
num_list = [65527,21907,27336,29526]

for i,csname in enumerate(path):
    csvname = csname
    df = pd.read_csv(csvname+'.csv', encoding="utf8")
    for clm in name:
        x1 = df.loc[:num_list[i]-1, clm]
        x1 = x1.as_matrix().astype('float16')

        t = range(0, len(x1) // fs)

        power, freqs, timeBins, img = plt.specgram(x1, NFFT=lenWindow, Fs=fs, noverlap=over_lap, window=hammWindow, cmap='gray', aspect="auto")

        np.save('specgram/' + csvname + '/' + clm+'_0.npy', power)

for i,csname in enumerate(path):
    csvname = csname
    df = pd.read_csv(csvname+'.csv', encoding="utf8")
    for clm in name:
        x1 = df.loc[num_list[i]:, clm]
        x1 = x1.as_matrix().astype('float16')

        t = range(0, len(x1) // fs)

        power, freqs, timeBins, img = plt.specgram(x1, NFFT=lenWindow, Fs=fs, noverlap=over_lap, window=hammWindow, cmap='gray', aspect="auto")

        np.save('specgram/' + csvname + '/' + clm+'_1.npy', power)



"""
        plt.axis([0, t[-1], 0, fs / 2])
        plt.xlabel("time[sec]")
        plt.ylabel("frequency[Hz]")
        plt.imshow()
        plt.savefig('specgram/' + csvname + '/' + clm+'_0')
"""


