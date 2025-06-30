#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% MODULOS + FUNCIONES
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import sympy as sp

from numpy import random
#-----
def mostrar_tiempo(xx, xxzp):
    
    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.plot(tt, xx)

    plt.subplot(1,2,2)
    plt.plot(ttn[bfrecn], xxzp[bfrecn])
    plt.show()

#-----
def mostrar_frecuencia(fft, fftzp):
    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.title('en db')
    plt.plot(ffn[bfrecn], 10*np.log10(2*np.abs(fftzp[bfrecn])**2))
    plt.legend([
            'con zero padding',
    #        'sin zero padding'
        ])

    plt.subplot(1,2,2)
    plt.title('fft')
    plt.plot(ff[bfrec], np.abs(fft[bfrec]))
    plt.plot(ffn[bfrecn], np.abs(fftzp[bfrecn]))
    plt.legend([
            'sin zero padding',
            'con zero padding'
        ])
    plt.show()
# %%
k = 5
N = 1000
fs = 1000

tt = np.arange(0, 1, 1/fs)
ttn = np.arange(0, k, 1/fs)
ff = np.arange(0, N)
ffn = np.arange(0, N, fs/(k*N))
zeros = np.zeros((k-1)*N)
bfrec = ff <= fs/2
bfrecn = ffn <= fs/2

# --boxcar ventana normal
boxcar = sig.windows.boxcar(N)
boxcarzp = np.concatenate([boxcar, zeros])

fft_boxcarzp = np.fft.fft(boxcarzp)/(N)
fft_boxcar = np.fft.fft(boxcar)/N

# --hamming, 
hamming = sig.windows.hamming(N)
hammingzp = np.concatenate([hamming, zeros])

fft_hamming = np.fft.fft(hamming)/N
fft_hammingzp = np.fft.fft(hammingzp)/N

# --blackman
blackman = sig.windows.blackman(N)
blackmanzp = np.concatenate([blackman, zeros])

fft_blackman = np.fft.fft(blackman)/N
fft_blackmanzp = np.fft.fft(blackmanzp)/N

# --bohman
bohman = sig.windows.bohman(N)
bohmanzp = np.concatenate([bohman, zeros])

fft_bohman = np.fft.fft(bohman)/N
fft_bohmanzp = np.fft.fft(bohmanzp)/N

# --flattop
flattop = sig.windows.flattop(N)
flattopzp = np.concatenate([flattop, zeros])

fft_flattop = np.fft.fft(flattop)/N
fft_flattopzp = np.fft.fft(flattopzp)/N

# --Hann
hann = sig.windows.hann(N)
hannzp = np.concatenate([hann, zeros])

fft_hann = np.fft.fft(hann)/N
fft_hannzp = np.fft.fft(hannzp)/N
# %% mostrar

#mostrar_tiempo(boxcar, boxcarzp)
#mostrar_frecuencia(fft_boxcar, fft_boxcarzp)

#mostrar_tiempo(hamming, hammingzp)
#mostrar_frecuencia(fft_hamming, fft_hammingzp)

mostrar_tiempo(blackman, blackmanzp)
mostrar_frecuencia(fft_blackman, fft_blackmanzp)

#mostrar_tiempo(bohman, bohmanzp)
#mostrar_frecuencia(fft_bohman, fft_bohmanzp)

#mostrar_tiempo(flattop, flattopzp)
#mostrar_frecuencia(fft_flattop, fft_flattopzp)

#mostrar_tiempo(hann, hannzp)
#mostrar_frecuencia(fft_hann, fft_hannzp)
