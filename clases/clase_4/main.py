import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from toolbox import *

N = 1000
fs = 1000
#-----------------------------#
#Ruido
U = 1
SnraDb = 10
Pa = 1
#sigmaCuadrado = 10*np.log10(-SnraDb) #Con esto no funciona
sigmaCuadrado = Pa/(10**(SnraDb/10))
Na = random.normal(0,np.sqrt(sigmaCuadrado),N)

#-----------------------------#
#mi señal
freq = 5
Amp = np.sqrt(2)
tt,xx = mi_funcion_sen(vmax=Amp, ff=freq, nn=N, fs = U*fs)
#tt,xx = mi_funcion_cuadrada(vmax=Amp, ff=freq, nn=N,fs=U*fs)
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

#------------------------------------#
#proceso inicial-creacion de la señal#
#------------------------------------#
plt.figure(figsize=(8,6))

plt.subplot(211)
plt.title(f'Seno con ruido de {freq} Hz')
plt.plot(tt, xa)

plt.subplot(212)
plt.title('FFT')
plt.plot(ff, np.abs(FF))

plt.tight_layout()
plt.show()

#---------------------------------------#
#segundo proceso-cuantizacion de la señal
#---------------------------------------#

#valores por defecto del adc, por asi decirlo
Vref = 5   #tension de referencia, ponele
n = 8       #resolucion
fc=0.8      #factor de carga
q = Vref/(2**(n-1))

xc = (xa*1/q)
xd = np.round(xc)
xf = xd*q


plt.figure(figsize=(8,6))
plt.plot(tt,xf)
