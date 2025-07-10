import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.io as sio

def banda(pw, ff):
    #normalizado a 1
    c_sum = np.cumsum(pw)/np.sum(pw)
    idx_s = np.argmax(np.isclose(c_sum, 0.95, atol=1e-3))
    idx_i = np.argmax(np.isclose(c_sum, 0.05, atol=1e-3))
    fp_s = int(ff[idx_s])
    fp_i = int(ff[idx_i])
    return idx_s, idx_i, fp_s, fp_i, c_sum

#%%
# Parámetros
fs = 1000  # Hz
#_____PASA BAJOS_____
#wp = 35
#ws = 45
#gpass = 1  # dB
#gstop = 40  # dB
#_____PASA BANDA_____
#creo que me equivoque aqui al usar el mismo rango para los dos filtros
wp = [1, 40]
ws = [0.1, 50]
gpass = 0.1
gstop = 20

#tipo = 'butter'
tipo = 'ellip'  #muy bueno
#tipo = 'cheby1'
#tipo = 'cheby2' #muy bueno

# Diseño del filtro
sos = sig.iirdesign(wp, ws, gpass, gstop, ftype=tipo, output='sos', fs=fs)

# Visualización de la respuesta en frecuencia
w, h = sig.sosfreqz(sos, worN=2000, fs=fs)

fase = np.unwrap(np.angle(h))
gd = -np.diff(fase)/np.diff(w)
w_mid = (w[:-1] + w[:1])/2

#Muestro
plt.figure(figsize=(12,6))
plt.plot(w, 20 * np.log10(abs(h)))
plt.title(f'Filtro Pasa Bajos IIR {tipo}')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Ganancia [dB]')
plt.grid(True)
#plt.axvline(wp, color='green', linestyle='--', label=f'fpass {wp}')
#plt.axvline(ws, color='red', linestyle='--', label=f'fstop {ws}')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.title('Retardo de Grupo')
plt.plot(w_mid, gd)
plt.ylabel('Retardo [muestras]')
plt.xlabel('Frecuencia [Hz]')
plt.grid(True)

#%% --------------------SEÑAL----------------------
archive= 'ECG_TP4.mat'

mat_struct = sio.loadmat(archive)
ecg_one_lead = mat_struct['ecg_lead'].flatten()
cant_muestras = len(ecg_one_lead)

ff, pw = sig.periodogram(ecg_one_lead, window='hamming', fs=fs)
idxs, idxi, fps, fpi, csum = banda(pw, ff)
db = 10*np.log10(2*np.abs(pw)**2)

#reginoes de interes
regs_interes = ( 
        np.array([2, 2.2]) *60*fs, # minutos a muestras
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([10, 10.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        np.array([18, 18.2]) *60*fs, # minutos a muestras
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )
for ii in regs_interes:
    
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
    filtrado_filt = sig.sosfilt(sos, ecg_one_lead[zoom_region])    
    filtrado_filtfilt = sig.sosfiltfilt(sos, ecg_one_lead[zoom_region])
    
    plt.figure(figsize=(12,6))
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, filtrado_filt, label='ECG-filt', linewidth=1)
    plt.plot(zoom_region, filtrado_filtfilt, label='ECG filt filt', linewidth=1)
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
            
    plt.show()

#muestro
#plt.figure(figsize=(12,6))
#plt.plot(ecg_one_lead)
#plt.plot(ff[idxi:idxs], np.abs(pw)[idxi:idxs])
#plt.plot(ff, csum)
#plt.plot(ff, db)
#plt.axvline(fps, color='green', linestyle='--', label=f'fs:{fps}')
#plt.axvline(fpi, color='red', linestyle='--', label=f'fi:{fpi}')
#plt.legend()
#plt.show()
