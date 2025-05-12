import numpy as np

#----------------#
#---CONSTANTES---#
#----------------#
N = 1000
fs = 1000

#------------------------------#
#---MIS FUNCIONES REALIZADAS---#
#------------------------------#
def cuantizar(Sr, Vf, B):
    q = Vf/(2**(B-1))
    Srq = (Sr*1/q)
    Sq = np.round(Srq)*q
    
    return Sq, q
    
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
