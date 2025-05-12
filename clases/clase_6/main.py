#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% MODULOS + FUNCIONES
import numpy as np
import matplotlib.pyplot as plt
import scipy as sig

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
    Srq = (Sr*1/q)
    Sq = np.round(Srq)*q
    
    return Sq, q
    
# %% ANALISIS


# cantidad de veces más densa que se supone la grilla temporal para tiempo "continuo"
over_sampling = 1
N_os = N*over_sampling
fs_os = fs*over_sampling

ts = 1/fs
df = fs/N
ts_os = 1/fs_os
df_os = fs_os/N_os
 
# Datos del ADC
B = 4 # bits
Vf = 2 # Volts
q = Vf/2**(B-1) # Volts

# datos del ruido
kn = 10
pot_ruido = ((q**2)/12) * kn # Watts (potencia de la señal 1 W)
Pn = pot_ruido

f0 = fs/N

#ruido analogico
#Snr = 10
#Ps = 1
#Pn = Ps*10**(-Snr/10)
#no me funciona, por alguna razon fallo en algunas cosas

tt, s = mi_funcion_sen(vmax=np.sqrt(2), nn=N, ff=f0)
tt_os, analog_sig = mi_funcion_sen(vmax=np.sqrt(2), ff=f0, nn=N_os, fss=fs_os)
n = random.normal(0, np.sqrt(Pn), N)
sr = s + n #señal contaminada con ruido
srq, _ = cuantizar(sr, Vf, B)
nq = srq - sr

#realizacion de la fft
ff = np.arange(0, N, df)
ff_os = np.arange(0, N_os, df_os)
bfrec = ff<=fs/2

ft_Srq = np.fft.fft(srq)
ft_As = np.fft.fft(analog_sig)
ft_SR = np.fft.fft(sr)
ft_Nn = np.fft.fft(n)
ft_Nq = np.fft.fft(nq)

#plt.plot(tt_os, analog_sig)
# %% Presentación gráfica de los resultados
plt.close('all')
 
plt.figure(1, figsize=(12,6))
plt.plot(tt, srq, lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\} $ (ADC out)')
plt.plot(tt, sr, linestyle=':', color='green',marker='o', markersize=3, markerfacecolor='none', markeredgecolor='green', fillstyle='none', label='$ s_R = s + n $  (ADC in)')
plt.plot(tt_os, analog_sig, color='orange', ls='dotted', label='$ s $ (analog)')
 
plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()
 
 
plt.figure(2, figsize=(12,6))
bfrec = ff <= fs/2
 
Nnq_mean = np.mean(np.abs(ft_Nq)**2)
nNn_mean = np.mean(np.abs(ft_Nn)**2)
 
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\} $ (ADC out)' )
plt.plot( ff_os[ff_os <= fs/2], 10* np.log10(2*np.abs(ft_As[ff_os <= fs/2])**2), color='orange', ls='dotted', label='$ s $ (analog)' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$ s_R = s + n $  (ADC in)' )
plt.plot( ff_os[ff_os <= fs/2], 10* np.log10(2*np.abs(ft_Nn[ff_os <= fs/2])**2), ':r')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB (piso analog.)'.format(10* np.log10(2* nNn_mean)) )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB (piso digital)'.format(10* np.log10(2* Nnq_mean)) )
plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()
# suponiendo valores negativos de potencia ruido en dB
plt.ylim((1.5*np.min(10* np.log10(2* np.array([Nnq_mean, nNn_mean]))),10))
 
 
plt.figure(3, figsize=(12,6))
bins = 10
plt.hist(nq, bins=bins)
plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))
