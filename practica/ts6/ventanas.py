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

Vmax = np.sqrt(2)
binf = N/4
shift = 1/2
freq = N/4 + shift
freq1 = N/4 - shift

tt, x0 = mi_funcion_sen(vmax = Vmax, ff = freq)
_ , x1 = mi_funcion_sen(vmax = Vmax, ff = freq1)

#xx = np.append(x0, np.zeros(9*N))
xx = np.concatenate([x0, np.zeros(9*N)])
# fft
ff = np.arange(0,N)
bfrec = ff<=fs/2

fft_x0 = np.fft.fft(x0)/N
fft_x1 = np.fft.fft(x1)/N
ttf = np.arange(0, 10, 1/fs)

fft_xx = np.fft.fft(xx)/(10*N)
ffx = np.arange(0, 10*N)
bfrecx = ffx<=fs/2
# %%
plt.figure(figsize=(13,6))

plt.subplot(1,2,1)
plt.title(f'corriendo el bien en +{shift}')
#plt.plot(ff[bfrec], 10*np.log10(2*np.abs(fft_x0[bfrec])**2), ':o')
#plt.plot(ff[bfrec], (np.zeros(N) - 13.5)[bfrec])

plt.plot(ttf, xx)

plt.subplot(1,2,2)
plt.title(f'corriendo el bin en -{shift}')
plt.plot(ff[bfrec], 10*np.log10(2*np.abs(fft_x1[bfrec])**2), ':o')
plt.plot(ff[bfrec], (np.zeros(N) - 13.5)[bfrec])


plt.figure(2, figsize=(13, 6))
plt.title('Realizando con un 10N por asi decirlo')

plt.plot(ffx[bfrecx], 10*np.log10(2*np.abs(fft_xx[bfrecx])**2), ':o')


plt.show()

