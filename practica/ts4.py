#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% MODULOS + FUNCIONES
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
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
# %% constantes (por asi decirlo)
over_sampling = 4
N_os = over_sampling*N
ts = 1/fs
df = fs/N
f0 = 1200
Vmax = np.sqrt(2)
bins = 10

# Datos del ADC
B = 4           #bits
Vf = 2          #Volts
q = Vf/2**(B-1) #Volts

#ruido digital teorico
kn = 10
pot_ruido = ((q**2)/12) * kn

#1. Señal original @ 1 kHz
#2. Oversampling → @ 4 kHz
#3. Decimación (reducción a 1 kHz)
#4. FFT para analizar y graficar → eje f = 0 a 500 Hz

# %% Analisis

#PARTE 1---- ananlisis del ruido
tt, s = mi_funcion_sen(vmax=Vmax, ff=f0)
n = random.normal(0, np.sqrt(pot_ruido), N)
sr = s + n                      #señal con ruido
srq, _ = cuantizar(sr, Vf, B)   #señal cuantizada
nq = srq - sr                   #error de cuantizacion

#PARTE 2 --- oversampling
tt_os, s_os = mi_funcion_sen(vmax=Vmax, ff=f0, nn=N_os, fss=over_sampling*fs)
n_os = random.normal(0, np.sqrt(pot_ruido), N_os)
sr_os = s_os + n_os
srq_os, _ = cuantizar(sr_os, Vf, B)
nq_os = srq_os - sr_os

srq_dos = srq_os[::over_sampling]       #decimacion (si es sin filtro: aliasing)

#realizacion de la fft
ff = np.arange(0, N, df)
bfrec = ff<=fs/2

ff_os = np.arange(0, N_os, df)
bfrec_os = ff_os <= fs/2

ft_S = np.fft.fft(s)/N
ft_Sr = np.fft.fft(sr)/N
ft_Srq = np.fft.fft(srq)/N
ft_Nn = np.fft.fft(n)/N
ft_Nq = np.fft.fft(nq)/N

nNn_mean = np.mean(np.abs(ft_Nn)**2)
nNq_mean = np.mean(np.abs(ft_Nq)**2)


ft_Sos = np.fft.fft(s_os)/N_os
ft_Sros = np.fft.fft(sr_os)/N_os
ft_Srqos = np.fft.fft(srq_os)/N_os
ft_Nnos = np.fft.fft(n_os)/N_os
ft_Nqos = np.fft.fft(nq_os)/N_os

nNn_mean_os = np.mean(np.abs(ft_Nnos)**2)
nNq_mean_os = np.mean(np.abs(ft_Nqos)**2)

ft_Srqdos = np.fft.fft(srq_dos)/N

# %% Muestra de resultados
plt.figure(1, figsize=(12,6))
plt.title('pisos de ruido')
plt.plot(ff_os[bfrec_os], 10*np.log10(2*np.abs(ft_Srqos[bfrec_os]**2)))
plt.plot(ff[bfrec], 10*np.log10(2*np.abs(ft_Srqdos[bfrec]**2)))
#plt.plot(ff[bfrec], 10*np.log10(2*np.abs(ft_Srq[bfrec])**2))

plt.legend([
        '$Srq_{os}$',
        '$Srq_{dos}$',
        #'$Srq$'
    ])

plt.show()
