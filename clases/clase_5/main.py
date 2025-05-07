import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from toolbox import mi_funcion_sen

N = 1000
fs = 1000


#-----------------------------#
U = 1
SnraDb = 10
Pa = 1
#sigmaCuadrado = 10*np.log10(-SnraDb) #Con esto no funciona
sigmaCuadrado = Pa/(10**(SnraDb/10))
Na = random.normal(0,np.sqrt(sigmaCuadrado),N)
#-----------------------------#
freq = 10
Amp = np.sqrt(2)
tt,xx = mi_funcion_sen(vmax=Amp, ff=freq, nn=N, fs = U*fs)
xa= xx + Na

FFxa = np.fft.fft(xa)[:N//2]*1/N #escalado
FFNa = np.fft.fft(Na)[:N//2]*1/N #escalado

FFxadb = 10*np.log10(np.abs(FFxa)**2)
FFNadb = 10*np.log10(np.abs(FFNa)**2)
#-----------------------------#
Bwa = U*fs/2
SnraCalculado = 10*np.log10((np.var(xa))/Bwa) #esta mal
#SnraCalculado = 10 * np.log10(np.var(xx) / np.var(nq))  # O error de cuantización

print(f'Valor de sigma cuadrado: {sigmaCuadrado}')
print(f'Valor de Snara calculado: {SnraCalculado}')
print(f'Potencia de la señal sin ruido: {np.var(xx)}')
print(f'Potencia de la señal con ruido: {np.var(xa)}')

#------------------------------------#
#proceso inicial-creacion de la señal#
#------------------------------------#

'''
plt.figure(figsize=(10,6))

plt.subplot(211)
plt.title(f'Seno con ruido de {freq} Hz')
plt.plot(tt, xa)

plt.subplot(212)
plt.title('FFT en db')
plt.plot(FFxa, label='fft xa')
plt.plot(FFNa, label='fft Na')
plt.legend()

plt.tight_layout()
plt.show()
'''
#---------------------------------------#
#segundo proceso-cuantizacion de la señal
#---------------------------------------#

#valores por defecto del adc, por asi decirlo
fc=0.8              #factor de carga
Vref = np.sqrt(2)   #tension de referencia
Vfc = fc*Vref       #Tension limite
n = 4               #resolucion
q = Vfc/(2**(n-1))  #paso de cuantizacion

xxq = np.round( (xx*1/q) ) * q

'''
plt.figure(figsize=(10,6))

plt.subplot(211)
plt.title(f'cuantizado a {n} bits')
plt.plot(tt,xxq8)

plt.subplot(212)
plt.title(f'cuantizado a {n4} bits')
plt.plot(tt,xxq4)

plt.tight_layout()
plt.show()
'''

print('------------------------------------')
print('-----------tercera parte------------')
print('------------------------------------')
nq = xxq - xx

plt.figure(figsize=(10,6))
plt.subplot(211)
plt.plot(tt, xx)
plt.plot(tt, xxq)
plt.plot(tt,nq)

plt.subplot(212)
plt.hist(nq)
#plt.plot(tt,nq)

plt.tight_layout()
plt.show()

# Potencia teórica del error para cuantización uniforme
Pteo = (q**2) / 12
# Potencia real (cuadrado medio del error)
Preal = np.mean(nq**2)

print(f'rango de q [{-q/2} :: {q/2}]')
print(f"Potencia teórica del error: {Pteo:.6f}")
print(f"Potencia real del error:    {Preal:.6f}")


#------------------------------------------------#
print('------------------------------------')
print('-----------cuarta parte------------')
print('------------------------------------')

Ps = np.mean(xx**2)
Pn = np.mean(nq**2)
SNR = 10 * np.log10(Ps / Pn)
print(f"SNR: {SNR:.2f} dB")

FFNq = np.fft.fft(nq)[:N//2]*1/N #fft escalado 
FFNadb = 10*np.log10(np.abs(FFNa)**2)
FFxadb = 10*np.log10(np.abs(FFxa)**2)
FFNqdb = 10*np.log10(np.abs(FFNq)**2)

plt.figure(figsize=(12,6))

plt.subplot(211)
plt.plot(FFxadb, label='xa')
plt.plot(FFNadb, label='na')
plt.legend()

plt.subplot(212)
plt.plot(FFNqdb, label='nq')
plt.legend()

plt.show()
#tengo que mostrar el xq tambien, de esta manera tengo que mostrar todo....
#en analogico quiero un ruido mucho menor,
#pero en digital quiero ampliar es decir un sobremuestreo donde pueda llegar a realizar muchas cosas








