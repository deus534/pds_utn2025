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
V = np.sqrt(2)
f0 = 250    #frecuencia central
df = 50     #separacion Hz
k = 2

ff = np.arange(0, N)
ffn = np.arange(0, N, fs/(k*N))
zeros = np.zeros((k-1)*N)
bfrec = ff <= fs/2
bfrecn = ffn <= fs/2

# mis funciones
tt, xx1 = mi_funcion_sen(V, ff=(f0+df/2), nn=k*N)
_ , xx2 = mi_funcion_sen(V, ff=(f0-df/2), nn=k*N)
xx = xx1 + xx2

# mis ventanas
blackman = np.concatenate([sig.windows.blackman(N), zeros])
boxcar = np.concatenate([sig.windows.boxcar(N), zeros])
hamming = np.concatenate([sig.windows.hamming(N), zeros])

vent = blackman
xx_vent = xx*vent
#fft
fft_xx = np.fft.fft(xx)/(k*N)
fft_xx_vent = np.fft.fft(xx_vent)/(k*N)

#VISUALIZACION
plt.figure(figsize=(12,6))
plt.plot(tt,xx)
plt.plot(tt,xx_vent)
plt.plot(tt,vent)
plt.show()

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.plot(ffn[bfrecn], np.abs(fft_xx[bfrecn]))
plt.plot(ffn[bfrecn], np.abs(fft_xx_vent[bfrecn]))

plt.subplot(1,2,2)
plt.plot(ffn[bfrecn], 10*np.log10(2*np.abs(fft_xx[bfrecn])**2))
plt.plot(ffn[bfrecn], 10*np.log10(2*np.abs(fft_xx_vent[bfrecn])**2))

plt.show()

