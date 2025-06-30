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
def matriz_sen(vmax, ff, fs=fs, nn=N):
    t = nn/fs
    tt = np.arange(0,t,1/fs)
    xx = []
    for i in range(len(ff)):
        xx.append(vmax*np.sin(2*np.pi*ff[i]*tt))
    return tt, xx
# %%

# 1. Efecto de las distintas ventanas
R = 10
a0 = np.sqrt(2)
f0 = fs/4
df = fs/N
k = 1
SnraDb = 3
sigma = 1/(10**(SnraDb/10))

# ----GENERACION DE SEÑALES----
vent = sig.windows.get_window('boxcar', N)
fr = random.uniform(-2,2,R)
na = random.normal(0, np.sqrt(sigma), N)
f1 = f0 + fr*df
w0 = 2*np.pi*f0
w1 = 2*np.pi*f1

tt, xx = matriz_sen(a0, w1)
_ , xxa = mi_funcion_sen(a0, ff=w0)

xxn = [xx[i] + na for i in range(R)]    #con ruido
xxf = [xx[i]*vent for i in range(R)]    #con ventaneo

# ----FFT----
ff = np.arange(0,N,fs/N)
bfrec = ff<=fs/2
fft_xxf = [np.fft.fft(xxf[i])/N for i in range(R)]
fft_xxa = np.fft.fft(xxa)/N

# ----estimador de amplitud----
amp_iesi = [np.abs(fft_xxf[i]) for i in range(R)]
fre_isei = [np.angle(fft_xxf[i]) for i in range(R)]

# ----calculo experimental-----
a_0 = np.abs(fft_xxa)
sa = np.mean(a_0) - a0      #sesgo
va = np.var(a_0)            #varianza

# ----aproximaciones----
mu_apx = [np.sum(amp_iesi[i])/N for i in range(R)]
sa_apx = [mu_apx[i] - a0 for i in range(R)]
va_apx = [(np.sum(amp_iesi[i]-mu_apx[i])**2)/N for i in range(R)]



# %%
plt.close('all')

plt.figure(figsize=(12,6))
for i in range(R):
    #plt.plot(tt, xxf[i])
    plt.plot(ff[bfrec], np.abs(fft_xxf[i][bfrec]))
#plt.xlim([w0-10, w0+10])

#plt.hist(mu_apx)
#plt.hist(sa_apx)
#plt.hist(va_apx)
plt.show()
