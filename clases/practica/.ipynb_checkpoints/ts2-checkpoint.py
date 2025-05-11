import numpy as np
import matplotlib.pyplot as plt
import sys

N = 1000
fs = 1000

#-------------------------------------------------------------------------------------#
#funciones sacadas de asys....
#Definición de la función ESCALÓN UNITARIO u(t)
u = lambda t: np.piecewise(t, t>=0, [1,0])

#Definición de la función RAMPA UNITARIA rho(t)
rho = lambda t: t*u(t)
#-------------------------------------------------------------------------------------#

def mi_funcion_sen(vmax = 1, dc = 0, ff = 1, ph = 0, nn = N, fs = fs):
    t = nn/fs
    tt = np.arange(0,t,1/fs)
    xx = vmax*np.sin(2*np.pi*ff*tt + ph) + dc

    return tt, xx

def mi_funcion_DFT( xx ):
    N = len(xx)
    n = np.arange(N)
    yy = np.zeros(N,dtype='complex128')
    for k in range(N):
        yy[k] = np.sum(xx * np.exp(-1j*2*np.pi*k*n/N))
    return yy
    
def mi_funcion_cuadrada(vmax=1, dc=0, ff=1, ph=0, nn=N, fs=fs):
    #me falta la parte de cuando tiene un desfase, que no se me ocurre por ahora
    tt = np.arange(0,nn/fs,1/fs)
    xx = np.zeros(fs)
    for i in range(fs):
        xx[i] = (vmax if (np.sin(2*np.pi*ff*i/fs)>=0) else -vmax ) + dc
    return tt, xx

def mi_funcion_triangular(vmax=1, dc=0, ff=1, ph=0, nn=N, fs=fs):
        

#-------------------------------------------------------------------------------------#
def generate_triangle_wave_samples( freq, sample_rate, Vpp, offset, duration=1):
    num_samples = int(sample_rate * duration)
    x = np.arange(0, num_samples)
    y = np.zeros( num_samples )
    amplitude = Vpp / 2
    samples_per_cycle = int(sample_rate / freq)
    for i in range( num_samples ):
        phase = i % samples_per_cycle
        if phase < samples_per_cycle / 2:  # Fase ascendente
            y[i] = 2 * amplitude * (phase / (samples_per_cycle / 2)) - amplitude + offset
        else:  # Fase descendente
            y[i] = 2 * amplitude * (1 - (phase - samples_per_cycle / 2) / (samples_per_cycle / 2)) - amplitude + offset
    return x,y

#-------------------------------------------------------------------------------------#
def test1():
    f = 1
    if len(sys.argv)>1:
        f = int(sys.argv[1])
    tt, xx = mi_funcion_sen(1,0,f)
    yy = mi_funcion_DFT(xx) 
    ff = np.arange(N)*fs/N

    plt.subplot(211)
    plt.title(f'seno de {f} Hz')
    plt.plot(tt,xx)

    plt.subplot(212)
    plt.title(f'DFT de xx')
    plt.plot(ff[:(N//2)],np.abs(yy[:(N//2)]))

    plt.tight_layout()
    plt.show()

def test2():
    freq = 2
    #tt,xx = mi_funcion_cuadrada(1,0,freq)
    tt,xx = mi_funcion_triangular(1,0,freq)
    plt.plot(tt,xx)
    plt.show()
if __name__=='__main__':
#    test1()
    test2()

