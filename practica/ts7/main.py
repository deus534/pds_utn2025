import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


fs = 10000
N = 10000
Vpp = np.sqrt(2)
f0 = fs/4
SnraDb = 3
sigma = 1/(10**(SnraDb/10))

tt = np.arange(0,1,1/fs)
xx = Vpp*np.sin(2*np.pi*f0*tt) + np.random.normal(0, np.sqrt(sigma), N)

def fun(vent):
    ftxx, pxx = sig.periodogram(xx, window=vent, scaling='spectrum')
    db_xx = 10*np.log10(2*np.abs(pxx/N)**2)
    return ftxx,pxx,db_xx

ft = np.fft.fft(xx)/N
db = 10*np.log10(2*np.abs(ft[fs//2:]))
fper, _, db_per = fun('boxcar')

fwelch, pwelch = sig.welch(xx, window='boxcar')
dbwelch = 10*np.log10(2*np.abs(pwelch/N)**2)

plt.figure(figsize=(12,6))

plt.plot((fper*N)[1:], db)
plt.plot((fper*N)[1:], db_per[1:])
plt.plot((fwelch*N)[1:], dbwelch[1:])

plt.legend([
        'fft',
        'periodograma',
        'welch'
    ])

plt.show()
