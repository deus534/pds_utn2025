#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
##############################
##### MODULOS, FUNCIONES #####
##############################
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
######################
##### PARAMETROS #####
######################
#terminado solo falta mostrar
over_sampling = 4
N_os = over_sampling*N
ts = 1/fs
df = fs/N
f0 = 600
Vmax = np.sqrt(2)
bins = 10

# Datos del ADC
B = 4           #bits
Vf = 2          #Volts
q = Vf/2**(B-1) #Volts

#ruido digital teorico
kn = 10
pot_ruido = ((q**2)/12) * kn

#(pasos a seguir)
#1. Señal original @ 1 kHz
#2. Oversampling → @ 4 kHz
#3. Filtrado paso bajo (frecuencia de corte ≤ 500 Hz)
#4. Decimación (reducción a 1 kHz)
#5. FFT para analizar y graficar → eje f = 0 a 500 Hz

# %%
#########################
##### DISEÑO FILTRO #####
#########################
ftran = 0.1
ripple = 0.5       #db
attenuation = 40   #db
ftype = "butter"
btype = "lowpass"

fc = 1/over_sampling - ftran/2
fst = 1/over_sampling + ftran/2
frecs = [ 0, 0, fc, fst, 1.0 ]
gains = [ 0, -ripple/2, -attenuation/2, -attenuation/2 ]    #db
gains = 10**(np.array(gains)/20)

# paso 1.
orderz, wcutofz = sig.buttord( fc, fst, ripple, attenuation, analog=False)

# paso 2.
numz, denz = sig.iirfilter(orderz, wcutofz, rp=ripple, rs=attenuation, btype=btype, analog=False, ftype=ftype)
sos = sig.iirfilter(orderz, wcutofz, rp=ripple, rs=attenuation, btype=btype, analog=False, ftype=ftype, output='sos')

my_digital_filter = sig.TransferFunction(numz, denz, dt=1/fs)
my_digital_filter_desc = 'butter' + '_ord_' + str(orderz) + '_digital'

# %%
####################
##### ANALISIS #####
####################
#PARTE 1 --- 
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

#PARTE 3 --- filtrado
srq_osfilt = sig.sosfilt(sos, srq_os)           #filtrado
srq_dosfilt = srq_osfilt[::over_sampling]       #decimacion
srq_dos = srq_os[::over_sampling]               #decimacion sin filtro (posible aliasing)

#PARTE 4 --- fft
ff = np.arange(0, N, df)
bfrec = ff<=fs/2

ff_os = np.arange(0, N_os, df)
bfrec_os = ff_os <= fs/2

ft_Srq = np.fft.fft(srq)/N
ft_Srqos = np.fft.fft(srq_os)/N_os
ft_Srqosfilt = np.fft.fft(srq_osfilt)/N_os
ft_Srqdos = np.fft.fft(srq_dos)/N
ft_Srqdosfilt = np.fft.fft(srq_dosfilt)/N

# %% 
#########################
##### VISUALIZACION #####
#########################

# 1. Filtro
npoints = 1000
w_nyq = 2*np.pi*fs/2
w, mag, _ = my_digital_filter.bode(npoints)

plt.figure(1, figsize=(12,6))
plt.cla()
plt.plot(w/w_nyq, mag, label=my_digital_filter_desc)
plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')
plt.gca().set_xlim([0, 1])
plt.legend()
plt.show()

# 2. Ruido
plt.figure(2, figsize=(12,6))
plt.title('pisos de ruido')
plt.plot(ff[bfrec], 10*np.log10(2*np.abs(ft_Srq[bfrec]**2)))
plt.plot(ff[bfrec], 10*np.log10(2*np.abs(ft_Srqdos[bfrec]**2)))
plt.plot(ff[bfrec], 10*np.log10(2*np.abs(ft_Srqdosfilt[bfrec]**2)))

plt.legend([
        '$Srq$',
        '$Srq_{dos}$',
        '$Srq_{dosfilt}$',
    ])

plt.show()