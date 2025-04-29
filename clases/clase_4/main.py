import numpy as np
from numpy import random
import matplotlib.pyplot as plt

#import scipy.fft as fft


N = 10000
fs = 10000

def mi_funcion_sen(vmax = 1, dc = 0, ff = 1, ph = 0, nn = N, fs = fs):
    t = nn/fs
    tt = np.arange(0,t,1/fs)
    xx = vmax*np.sin(2*np.pi*ff*tt + ph) + dc

    return tt, xx

#-----------------------------#
#Ruido
U = 1
SnraDb = 10
Pa = 1
#sigmaCuadrado = 10*np.log10(-SnraDb)
sigmaCuadrado = Pa/(10**(SnraDb/10))
Na = random.normal(0,np.sqrt(sigmaCuadrado),N)

#-----------------------------#
#mi señal
freq = 5
Amp = np.sqrt(2)
tt,xx = mi_funcion_sen(vmax=Amp, ff=freq, fs = U*fs)
xa= xx + Na #sumo el ruido

FF = np.fft.fft(xa)[:N//2]
ff = np.arange(0,N/2)

#-----------------------------#
Bwa = U*fs/2
SnraCalculado = 10*np.log10((np.var(xa))/Bwa)
print(f'Valor de sigma cuadrado: {sigmaCuadrado}')
print(f'Valor de Snara calculado: {SnraCalculado}')
print(f'Potencia de la señal sin ruido: {np.var(xx)}')
print(f'Potencia de la señal con ruido: {np.var(xa)}')

#-----------------------------#
plt.figure(figsize=(8,6))

plt.subplot(211)
plt.title(f'Seno con ruido de {freq} Hz')
plt.plot(tt, xa)

plt.subplot(212)
plt.title('FFT')
plt.plot(ff, np.abs(FF))

plt.tight_layout()