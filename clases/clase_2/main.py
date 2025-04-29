import numpy as np
import matplotlib.pyplot as plt

N = 1000
fs = 1000

def mi_funcion_sen(vmax = 1, dc = 0, ff = 1, ph = 0, nn = N, fs = fs):
    t = nn/fs
    tt = np.arange(0,t,1/fs)
    xx = vmax*np.sin(2*np.pi*ff*tt + ph) + dc

    return tt, xx


f0 = 0.5*fs
tt, xx = mi_funcion_sen( 1,0,f0,0)

print(f"nyquist: {fs/2}")

plt.title(f"seno de {f0} HZ, con fs {fs}")
plt.grid()
plt.plot(tt,xx)
plt.show()

plt.close('all')