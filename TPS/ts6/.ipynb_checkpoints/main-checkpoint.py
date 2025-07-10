# %% MODULOS + FUNCIONES
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import sympy as sp
from scipy.fft import fft, fftshift

from numpy import random

fs = 1000
N = 1000
def mi_funcion_sen(vmax = 1, dc = 0, ff = 1, ph = 0, nn = N, fss = fs):
    t = nn/fss
    #tt = np.arange(0,t,1/fss)
    tt = np.linspace(0, t, nn, endpoint=False)
    xx = vmax*np.sin(2*np.pi*ff*tt + ph) + dc

    return tt, xx

def cuantizar(Sr, Vf, B):
    q = Vf/(2**(B-1))
    Srq = Sr*1/q
    Sq = np.round(Srq)*q
    
    return Sq, q
# %%
# ++++++++++++++PRIMERA PARTE+++++++++++++++++
# 1. analisis que siempre realizabamos (fft,etc)
# 2. efecto del ventaneo( frecc=500+0.5 )
# 3. efecto de la eleccion de la ventana (x[n]*vent[n])
# 4. fuga, extension espectral (leakage, spread)
#   - Al realizar un ventaneo llegamos a tener un minimo de separacion
#   entre tonos (por asi decrilo) que esto depende de la ventana que 
#   estemos utilizando, sino las señales se solapan
#   - Al aumentar el N de la ventana, el ancho del lobulo principal va
#   disminuyendo
#   -> LEAKAGE
#   . El ventaneo agrega energia fuera de banda, lo que actua como piso
#   de ruido para la medicion espectral.
#   . Creo que lo que se ve es que a medida que aumentas el N de la ventana
#   podes evitar el LEAKAGE, si tenes como que bajo N, el LEAKAGE se come uno
#   de tus tonos, (ejemplo leakage.py)
# 5. efecto del sampling
#   . lo que vimos en la clase 8. lo que se pasa no

#+++++++++++++++++++SEGUNDA PARTE+++++++++++++++++++
# metodo parametrico de estimacion espectral
# %%

# 1. Efecto de las distintas ventanas
Vmax = np.sqrt(2)
freq = 150
k = 2

tt = np.arange(0, 1, 1/fs)
ttn = np.arange(0, k, 1/fs)
ff = np.arange(0, N)
ffn = np.arange(0, N, fs/(k*N))
zeros = np.zeros((k-1)*N)
bfrec = ff <= fs/2
bfrecn = ffn <= fs/2

# ----GENERACION DE LAS SEÑALES----
tt, xx = mi_funcion_sen(Vmax, nn=k*N, ff=freq)
blackman = np.concatenate([sig.windows.blackman(N), zeros])
boxcar = np.concatenate([sig.windows.boxcar(N), zeros])
hamming = np.concatenate([sig.windows.hamming(N), zeros])

# +++VENTANEO+++
vent = hamming
xx_vent = xx * vent

# +++FFT+++
fft_vent = np.fft.fft(vent)/(k*N)
fft_xx = np.fft.fft(xx)/(k*N)
fft_xx_vent = np.fft.fft(xx_vent)/(k*N)

# ++++++++++VISUALIZACION++++++++
plt.figure(figsize=(12,6))

plt.plot(tt, xx)
plt.plot(tt, xx_vent)
plt.plot(tt, vent)
plt.legend([
        'signal',
        'window',
        'signal window'
    ])

plt.show()

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.plot(ffn[bfrecn], np.abs(fft_xx[bfrecn]))
plt.plot(ffn[bfrecn], np.abs(fft_vent[bfrecn]))
plt.plot(ffn[bfrecn], np.abs(fft_xx_vent[bfrecn]))
plt.legend([
        'signal',
        'window',
        'signal window'
    ])

plt.subplot(1,2,2)
plt.plot(ffn[bfrecn], 10*np.log10(2*np.abs(fft_xx[bfrecn])**2))
plt.plot(ffn[bfrecn], 10*np.log10(2*np.abs(fft_vent[bfrecn])**2))
plt.plot(ffn[bfrecn], 10*np.log10(2*np.abs(fft_xx_vent[bfrecn])**2))
plt.legend([
        'signal',
        'window',
        'signal window'
    ])

plt.show()
