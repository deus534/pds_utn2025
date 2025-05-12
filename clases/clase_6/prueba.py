#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
ts = 1/fs
df = fs/N
f0 = fs/N
Vmax = np.sqrt(2)
bins = 10
 
# Datos del ADC
B = 4           #bits
Vf = 2          #Volts
q = Vf/2**(B-1) #Volts

#ruido digital teorico
kn = 10
pot_ruido = ((q**2)/12) * kn

#ruido analogico
Snr = 10     #dB
Ps = 1      #Potencia señal
Pn = Ps/(10**(Snr/10)) 

tt, s = mi_funcion_sen(vmax=Vmax, ff=f0)
n = random.normal(0, np.sqrt(Pn), N)
sr = s + n                      #señal con ruido
srq, _ = cuantizar(sr, Vf, B)   #señal cuantizada
nq = srq - sr                   #error de cuantizacion

#realizacion de la fft
ff = np.arange(0, N, df)
bfrec = ff<=fs/2

ft_S = np.fft.fft(s)/N
ft_Sr = np.fft.fft(sr)/N
ft_Srq = np.fft.fft(srq)/N
ft_Nn = np.fft.fft(n)/N
ft_Nq = np.fft.fft(nq)/N

nNn_mean = np.mean(np.abs(ft_Nn)**2)
nNq_mean = np.mean(np.abs(ft_Nq)**2)
# %%

plt.figure(1, figsize=(12,6))
plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V y q = {:3.5f} V'.format(B, Vf, q))

plt.plot(tt, s, "blue", zorder=3 )
plt.plot(tt, sr, "purple", zorder=2)
plt.plot(tt, srq, "red", zorder=1, alpha=0.9)
plt.legend(["original S", "con ruido", "cuantizada"])

plt.tight_layout()
plt.show()

#--------------#
plt.figure(2, figsize=(12,6))
plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V y q = {:3.5f} V'.format(B, Vf, q))

plt.plot(ff[bfrec], 10*np.log10(2*np.abs(ft_Sr[bfrec])**2))
plt.plot(ff[bfrec], 10*np.log10(2*np.abs(ft_Srq[bfrec])**2))
plt.plot(ff[bfrec], 10*np.log10(2*np.abs(ft_Nn[bfrec])**2), "g", alpha=0.7)
plt.plot(ff[bfrec], 10*np.log10(2*np.abs(ft_Nq[bfrec])**2), "c", alpha=0.7)
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--g', zorder=6)
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNq_mean, nNq_mean]) ), '--c', zorder=5)

plt.legend(["$ Sr $",
            "$ Srq $",
            "$ n $",
            "$ nq $",
            '$ \overline{n} = $' + '{:3.1f} dB (piso analog.)'.format(10* np.log10(2* nNn_mean)),
            '$ \overline{n_Q} = $' + '{:3.1f} dB (piso digital)'.format(10* np.log10(2* nNq_mean))])

plt.tight_layout()
plt.show()

#-------------#
plt.figure(figsize=(12,6))
plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V y q = {:3.5f} V'.format(B, Vf, q))

plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
plt.hist( nq )

plt.show()
plt.tight_layout()
