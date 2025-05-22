#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% MODULOS + FUNCIONES
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import sympy as sp

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
    
# %% ANALISIS
over_sampling = 4
N_os = over_sampling*N
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
#Usamos el ruido digital teorico para ponerlo como que 10db por encima, igual o 10 db por debajo
#del ruido digital.
#Snr = 10     #dB
#Ps = 1      #Potencia señal
#Pn = Ps/(10**(Snr/10)) 

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

# %% diseño del filtro 

ftran = 0.1
ripple = 0.5       #db
attenuation = 40   #db

fc = 1/over_sampling - ftran/2
fst = 1/over_sampling + ftran/2
frecs = [ 0, 0, fc, fst, 1.0 ]
gains = [ 0, -ripple/2, -attenuation/2, -attenuation/2 ]    #db
gains = 10**(np.array(gains)/20)

# paso 1.
orderz, wcutofz = sig.buttord( fc, fst, ripple, attenuation, analog=False)

# paso 2.
numz, denz = sig.iirfilter(orderz, wcutofz, rp=ripple, rs=attenuation, btype="lowpass", analog=False, ftype="butter")

my_digital_filter = sig.TransferFunction(numz, denz, dt=1/fs)
my_digital_filter_desc = 'butter' + '_ord_' + str(orderz) + '_digital'

# Plantilla de diseño

plt.figure(1)
plt.cla()

npoints = 1000
w_nyq = 2*np.pi*fs/2

w, mag, _ = my_digital_filter.bode(npoints)
plt.plot(w/w_nyq, mag, label=my_digital_filter_desc)

plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

plt.gca().set_xlim([0, 1])

#plot_plantilla(filter_type = filter_type , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
plt.legend()

# %%
"""
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

#plt.plot(ff[bfrec], 10*np.log10(2*np.abs(ft_Sr[bfrec])**2))
#plt.plot(ff[bfrec], 10*np.log10(2*np.abs(ft_Srq[bfrec])**2))

plt.plot(ff[bfrec], 10*np.log10(2*np.abs(ft_Nn[bfrec])**2))
plt.plot(ff[bfrec], 10*np.log10(2*np.abs(ft_Nq[bfrec])**2))
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--g', zorder=6)
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNq_mean, nNq_mean]) ), '--c', zorder=5)

plt.plot(ff_os[ff_os <= over_sampling*fs/2], 10*np.log10(2*np.abs(ft_Srqos[ff_os <= over_sampling*fs/2]**2)))
#plt.plot(ff_os[bfrec_os], 10*np.log10(2*np.abs(ft_Srqos[bfrec_os])**2))
plt.plot(ff_os[bfrec_os], 10*np.log10(2*np.abs(ft_Nnos[bfrec_os])**2), zorder=4)
plt.plot(ff_os[bfrec_os], 10*np.log10(2*np.abs(ft_Nqos[bfrec_os])**2), zorder=3)
plt.plot( np.array([ ff_os[bfrec_os][0], ff_os[bfrec_os][-1] ]), 10* np.log10(2* np.array([nNn_mean_os, nNn_mean_os]) ), '--g', zorder=6)
plt.plot( np.array([ ff_os[bfrec_os][0], ff_os[bfrec_os][-1] ]), 10* np.log10(2* np.array([nNq_mean_os, nNq_mean_os]) ), '--c', zorder=5)

plt.legend([
            #"$ Sr $",
            #"$ Srq $",
            "$ n $",
            "$ nq $",
            '$ \overline{n} = $' + '{:3.1f} dB (piso analog.)'.format(10* np.log10(2* nNn_mean)),
            '$ \overline{n_Q} = $' + '{:3.1f} dB (piso digital)'.format(10* np.log10(2* nNq_mean)),
            '$ Srq\_os $',
            #'$ Srq\_os $',
            '$ n\_os $',
            '$ nq\_os $',
            '$ \overline{n\_os} = $' + '{:3.1f} dB (piso analog.)'.format(10* np.log10(2* nNn_mean_os)),
            '$ \overline{n\_os_{Q}} = $' + '{:3.1f} dB (piso digital)'.format(10* np.log10(2* nNq_mean_os)),
            ])

plt.tight_layout()
plt.show()

#-------------#
plt.figure(figsize=(12,6))
plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V y q = {:3.5f} V'.format(B, Vf, q))

plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
plt.hist( nq )

plt.show()
plt.tight_layout()
"""