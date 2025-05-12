#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#cosas que puedo aportar, por ejemplo lo que me dice copilot.
# %% 
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

N = 1000
fs = 1000
def mi_funcion_sen(vmax = 1, dc = 0, ff = 1, ph = 0, nn = N, fs = fs):
    t = nn/fs
    tt = np.arange(0,t,1/fs)
    xx = vmax*np.sin(2*np.pi*ff*tt + ph) + dc

    return tt, xx

def cuantizar(Sr, Vf, B):
    q = Vf/(2**(B-1))
    Srq = (Sr*1/q)
    Sq = np.round(Srq)*q
    
    return Sq, q
    

# %% DECLARACION DE CONSTANTES
f0 = fs/N
Vf = 2
B = 16
bins = 10

tt, Sr = mi_funcion_sen(ff=f0)
Sq, q = cuantizar(Sr, Vf, B)
e = Sq - Sr #error de cuantizacion 

# %% MUESTRO DE DATOS

print(f"media de error {np.mean(e)}")
print(f"media de varianza teorica: {q**2/12}")
print(f"media de varianza real: {np.var(e)}")
print(f"Correlacion: {np.corrcoef(Sr, e)[0, 1]}")

# %% PLOTEO - GRAFICACION

plt.figure(figsize=(12,6))

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V y q = {:3.5f} V'.format(B, Vf, q))
plt.plot(tt, Sr, label='cuantiza')
plt.plot(tt, Sq, label='normal')
plt.legend()

plt.tight_layout()

#distribucion uniforme
plt.figure(figsize=(12,6))
plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
plt.hist(e)


plt.figure(figsize=(12,6))
plt.scatter(Sr, e, alpha=0.5)
plt.xlabel("Señal Original")
plt.ylabel("Error de Cuantización")
plt.title("Gráfico de Dispersión: Señal vs. Error")
plt.grid(True)
plt.show()

