import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from scipy import signal as sig
from scipy.io.wavfile import write


fs_ecg = 1000
win = 'hamming'
archive_wav = 'la cucaracha.wav'
#archive_wav = 'prueba psd.wav'
#archive_wav = 'silbido.wav'
archive_mat = 'ECG_TP4.mat'
#archive_npy = 'ecg_sin_ruido.npy'
archive_npy = 'ppg_sin_ruido.npy'

#++++++++++++++++++++++++++++++++++#
#CARGADO DE DATOS#
#++++++++++++++++++++++++++++++++++#

#.wav
fs_wav, wav_data = sio.wavfile.read(archive_wav)
#.mat
mat_struct = sio.loadmat('ECG_TP4.mat')
patter1 = mat_struct['heartbeat_pattern1'].flatten()
patter2 = mat_struct['heartbeat_pattern2'].flatten()
#.npy
npy_data = np.load('ecg_sin_ruido.npy')


average_wel = 'mean'
#average_wel = 'media'
L_wav = fs_wav//10           #N per seg
L_ecg = fs_ecg//10
#L_wav = None
#L_ecg = None
#++++++++++++++++++++++++++++++++++#
#ANALISIS#
#++++++++++++++++++++++++++++++++++#

ff_wav_per, pw_wav_per = sig.periodogram(wav_data, window=win)
ff_wav_wel, pw_wav_wel = sig.welch(wav_data, window=win, nfft=(len(wav_data)))

db_wav_per = 10*np.log10(2*pw_wav_per**2)
db_wav_wel = 10*np.log10(2*pw_wav_wel**2)

# normalizo a frecuencia
ff_wav_per = ff_wav_per*fs_wav
ff_wav_wel = ff_wav_wel*fs_wav

#.mat
ff_patter1_per, pw_patter1_per = sig.periodogram(patter1, window=win)
ff_patter1_wel, pw_patter1_wel = sig.welch(patter1, window=win)

db_patter1_per = 10*np.log10(2*pw_patter1_per**2)
db_patter1_wel = 10*np.log10(2*pw_patter1_wel**2)


ff_patter2_per, pw_patter2_per = sig.periodogram(patter2, window=win)
ff_patter2_wel, pw_patter2_wel = sig.welch(patter2, window=win)

db_patter2_per = 10*np.log10(2*pw_patter2_per**2)
db_patter2_wel = 10*np.log10(2*pw_patter2_wel**2)

#.npy
ff_npy_per, pw_npy_per = sig.periodogram(npy_data, window=win)
ff_npy_wel, pw_npy_wel = sig.welch(npy_data, window=win, average=average_wel, nperseg=L_ecg)

db_npy_per = 10*np.log10(2*pw_npy_per**2)
db_npy_wel = 10*np.log10(2*pw_npy_wel**2)

ff_npy_per = ff_npy_per*fs_wav
ff_npy_wel = ff_npy_wel*fs_wav

#++++++++++++++++++++++++++++++++++#
#2do analisis#
#++++++++++++++++++++++++++++++++++#

#normalizado a 1
#c_sum = np.cumsum(pw_wav_wel)/np.sum(pw_wav_wel)
c_sum = np.cumsum(pw_wav_per)/np.sum(pw_wav_per)
freq_sup = 0
freq_inf = 0

for i in range(len(c_sum)):
    if (abs(c_sum[i]-0.95)<1e-3):
        freq_sup = i
        break
for i in range(len(c_sum)):
    if (abs(c_sum[i]-0.05)<1e-3):
        freq_inf = i
        break


#++++++++++++++++++++++++++++++++++#
#VISUALIZACION#
#++++++++++++++++++++++++++++++++++#
#potencia señal
plt.figure(figsize=(12,6))
plt.title(f'potencia señal - {archive_wav}')
plt.plot(c_sum)
plt.plot(freq_sup, 0.95, ':x')
plt.plot(freq_inf, 0.05, ':x')
plt.show()

#wav
plt.figure(figsize=(12, 6))
plt.title(f'{archive_wav} - {win} - L: {L_wav} - average: {average_wel}')

plt.subplot(2,1,1)
#plt.plot(wav_data)
plt.plot(ff_wav_per[freq_inf: freq_sup], db_wav_per[freq_inf:freq_sup])
#plt.plot(ff_wav_wel[freq_inf: freq_sup], db_wav_wel[freq_inf:freq_sup])
plt.legend([
        'periodogram',
#        'welch'
    ])

plt.subplot(2,1,2)
plt.plot(ff_wav_per, db_wav_per)
#plt.plot(ff_wav_wel, db_wav_wel)
plt.plot(freq_sup, 0, ':o')
plt.plot(freq_inf, 0, ':o')
plt.legend([
    'periodogram',
#    'welch'
])

plt.show()

#----------.npy-----------
#plt.figure(figsize=(12,6))
#plt.title(f'{archive_npy} - {win} - L: {L_ecg} - average: {average_wel}')

#plt.subplot(1,2,1)
#plt.plot(npy_data)

#plt.subplot(1,2,2)
#plt.plot(ff_npy_per, db_npy_per)
#plt.plot(ff_npy_wel, db_npy_wel)
#plt.legend([
#        'periodograma',
#        'welch'
#    ])
#plt.show()
'''
plt.figure(figsize=(12,6))
plt.title(f'{archive_mat} - patter1')

plt.subplot(1,2,1)
plt.plot(patter1)

plt.subplot(1,2,2)
plt.plot(ff_patter1_per, db_patter1_per)
plt.plot(ff_patter1_wel, db_patter1_wel)
plt.legend([
        'periodogram',
        'welch'
    ])
plt.show()


plt.figure(figsize=(12,6))
plt.title(f'{archive_mat} - patter2')

plt.subplot(1,2,1)
plt.plot(patter2)

plt.subplot(1,2,2)
plt.plot(ff_patter2_per, db_patter2_per)
plt.plot(ff_patter2_wel, db_patter2_wel)
plt.legend([
        'periodogram',
        'welch'
    ])
plt.show()
'''