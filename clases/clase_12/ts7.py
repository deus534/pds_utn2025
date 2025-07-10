import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from scipy import signal as sig



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


#++++++++++++++++++++++++++++++++++#
#ANALISIS#
#++++++++++++++++++++++++++++++++++#
#.wav
ff_wav_per, pw_wav_per = sig.periodogram(wav_data, window=win)
ff_wav_wel, pw_wav_wel = sig.welch(wav_data, window=win, nfft=(len(wav_data)))
db_wav_per = 10*np.log10(2*pw_wav_per**2)
db_wav_wel = 10*np.log10(2*pw_wav_wel**2)

#.mat
ff_patter1_per, pw_patter1_per = sig.periodogram(patter1, window=win)
ff_patter1_wel, pw_patter1_wel = sig.welch(patter1, window=win, nfft=len(patter1))
db_patter1_per = 10*np.log10(2*pw_patter1_per**2)
db_patter1_wel = 10*np.log10(2*pw_patter1_wel**2)

ff_patter2_per, pw_patter2_per = sig.periodogram(patter2, window=win)
ff_patter2_wel, pw_patter2_wel = sig.welch(patter2, window=win, nfft=len(patter2))
db_patter2_per = 10*np.log10(2*pw_patter2_per**2)
db_patter2_wel = 10*np.log10(2*pw_patter2_wel**2)

#.npy
ff_npy_per, pw_npy_per = sig.periodogram(npy_data, window=win)
ff_npy_wel, pw_npy_wel = sig.welch(npy_data, window=win, nfft=len(npy_data))
db_npy_per = 10*np.log10(2*pw_npy_per**2)
db_npy_wel = 10*np.log10(2*pw_npy_wel**2)

# normalizo a frecuencia
ff_wav_per = ff_wav_per*fs_wav
ff_wav_wel = ff_wav_wel*fs_wav
ff_npy_per = ff_npy_per*fs_ecg
ff_npy_wel = ff_npy_wel*fs_ecg

#++++++++++++++++++++++++++++++++++#
#2do analisis#
#++++++++++++++++++++++++++++++++++#
def banda(pw, ff):
    #normalizado a 1
    c_sum = np.cumsum(pw)/np.sum(pw)
    idx_s = np.argmax(np.isclose(c_sum, 0.95, atol=1e-3))
    idx_i = np.argmax(np.isclose(c_sum, 0.05, atol=1e-3))
    fp_s = int(ff[idx_s])
    fp_i = int(ff[idx_i])
    return idx_s, idx_i, fp_s, fp_i, c_sum

#++++++++++++++++++++++++++++++++++#
#VISUALIZACION#
#++++++++++++++++++++++++++++++++++#
def graficar(ff1, ff2, pw1, pw2, db1, db2, archive):
    idx_s1, idx_i1, fp_s1, fp_i1, csum1 = banda(pw1, ff1)
    idx_s2, idx_i2, fp_s2, fp_i2, csum2 = banda(pw2, ff2)
    name = archive[:archive.rindex('.')]
    #potencia señal
    plt.figure(figsize=(12,6))
    
    plt.subplot(1,2,1)
    plt.suptitle(f'Potencia normalizada --- {archive} - {win}')
    plt.title('Periodograma')
    plt.plot(ff1, csum1, label='Periodograma')
    plt.plot([fp_s1, fp_s1], [0,1], label=f'{fp_s1}')
    plt.plot([fp_i1, fp_i1], [0,1], label=f'{fp_i1}')
    plt.xlabel('frecuencia')
    plt.ylabel('Amp normalizada')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    plt.subplot(1,2,2)
    plt.title('Welch')
    plt.plot(ff2, csum2, label='welch')
    plt.plot([fp_s2, fp_s2], [0,1], label=f'{fp_s2}')
    plt.plot([fp_i2, fp_i2], [0,1], label=f'{fp_i2}')
    plt.xlabel('frecuencia')
    plt.ylabel('Amp normalizada')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{name}_potencia.png')
    plt.show()

    #grafico en db
    plt.figure(figsize=(12, 6))
    plt.suptitle(f'{archive} - {win}')
    
    plt.subplot(2,2,1)
    plt.title('banda recortada - periodograma')
    plt.plot(ff1[idx_i1: idx_s1], db1[idx_i1:idx_s1])
    plt.xlabel('frecuencia')
    plt.ylabel('db')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    plt.subplot(2,2,2)
    plt.title('banda ancha - periodograma')
    plt.plot(ff1, db1)
    plt.plot([fp_s1, fp_s1], [0,min(db1)], label=f'$f_s$={fp_s1}')
    plt.plot([fp_i1, fp_i1], [0,min(db1)], label=f'$f_i$={fp_i1}')
    plt.xlabel('frecuencia')
    plt.ylabel('db')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.subplot(2,2,3)
    plt.title('banda recortada - welch')
    plt.plot(ff2[idx_i2:idx_s2], db2[idx_i2:idx_s2])
    plt.xlabel('frecuencia')
    plt.ylabel('db')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    plt.subplot(2,2,4)
    plt.title('banda ancha - welch')
    plt.plot(ff2, db2)
    plt.plot([fp_s2, fp_s2], [0,min(db2)], label=f'$f_s$={fp_s2}')
    plt.plot([fp_i2, fp_i2], [0,min(db2)], label=f'$f_i$={fp_i2}')
    plt.xlabel('frecuencia')
    plt.ylabel('db')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{name}_db.png')
    plt.show()


    plt.figure(figsize=(12,6))
    plt.suptitle(f'{archive} - {win}')
    plt.subplot(2,2,1)
    plt.title('banda recortada - periodograma')
    plt.plot(ff1[idx_i1:idx_s1], pw1[idx_i1:idx_s1])
    plt.xlabel('frecuencia')
    plt.ylabel('magnitud')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    plt.subplot(2,2,2)
    plt.title('banda ancha - periodograma')
    plt.plot(ff1, pw1)
    plt.plot([fp_s1, fp_s1], [0,max(pw1)], label=f'$f_s$={fp_s1}')
    plt.plot([fp_i1, fp_i1], [0,max(pw1)], label=f'$f_i$={fp_i1}')
    plt.xlabel('frecuencia')
    plt.ylabel('magnitud')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.subplot(2,2,3)
    plt.title('banda recortada - welch')
    plt.plot(ff2[idx_i2:idx_s2], pw2[idx_i2:idx_s2])
    plt.xlabel('frecuencia')
    plt.ylabel('magnitud')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    plt.subplot(2,2,4)
    plt.title('banda ancha - welch')
    plt.plot(ff2, pw2)
    plt.plot([fp_s2, fp_s2], [0,max(pw2)], label=f'$f_s$={fp_s2}')
    plt.plot([fp_i2, fp_i2], [0,max(pw2)], label=f'$f_i$={fp_i2}')
    plt.xlabel('frecuencia')
    plt.ylabel('magnitud')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{name}_fft.png')
    plt.show()

#graficar(ff_wav_per, ff_wav_wel, pw_wav_per, pw_wav_wel, db_wav_per, db_wav_wel, archive_wav)
graficar(ff_npy_per, ff_npy_wel, pw_npy_per, pw_npy_wel, db_npy_per, db_npy_wel, archive_npy)
#graficar(ff_patter1_per, ff_patter1_wel, pw_patter1_per, pw_patter1_wel, db_patter1_per, db_patter1_wel, archive_mat)

