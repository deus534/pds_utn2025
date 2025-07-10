import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

fs = 1000
N = 1000
k = 200
f0 = fs/4
SnraDb = 3
sigma = 1/(10**(SnraDb/10))
a0 = np.sqrt(2)

def analisis(vent):
    fr = np.random.uniform(-2,2,k)
    win = sig.windows.get_window(vent, N)
    f1 = f0 + fr

    tt = np.arange(0, N/fs, 1/fs)
    ff = np.arange(0, N) 
    bfrec = ff<fs/2
    
    xx = [(a0*np.sin(2*np.pi*f1[i]*tt) + np.random.normal(0, np.sqrt(sigma), N))*win for i in range(k)]
    ff_xx = [np.fft.fft(xx[i])/N for i in range(k)]
    #db_xx = [(10*np.log10(np.abs(ff_xx[i])**2)) for i in range(k)]
    
    a_i = []
    f_i = []
    for i in range(k):
        idx = np.argmax(np.abs(ff[bfrec]-f1[i]))
        a_i.append(2*np.abs(ff_xx[i][idx]))
        f_i.append(ff[np.argmax(np.abs(ff_xx[i][bfrec]))])
    mu_a = np.mean(a_i)
    s_a = mu_a - a0
    v_a = np.mean((a_i - mu_a)**2)
    
    mu_f = np.mean(f_i)
    s_f = mu_f - f0
    v_f = np.mean((f_i - mu_f)**2)
    return a_i, f_i, [mu_a, s_a, v_a], [mu_f, s_f, v_f]

values = {
        'boxcar': analisis('boxcar'),
        'flattop': analisis('flattop'),
        'blackman': analisis('blackman'),
        'bohman': analisis('bohman')
    }

plt.figure(figsize=(12,6))
plt.hist(values['boxcar'][0], alpha=0.7)
plt.hist(values['flattop'][0], alpha=0.7)
plt.hist(values['blackman'][0], alpha=0.7)
plt.hist(values['bohman'][0], alpha=0.7)
plt.legend([
        'boxcar',
        'flattop',
        'blackman',
        'bohman'
    ])
plt.savefig(f'histograma_normal_{SnraDb}DB.png')
plt.show()

plt.figure(figsize=(12,6))
plt.hist(values['boxcar'][1], alpha=0.7)
plt.hist(values['flattop'][1], alpha=0.7)
plt.hist(values['blackman'][1], alpha=0.7)
plt.hist(values['bohman'][1], alpha=0.5)
plt.legend([
        'boxcar',
        'flattop',
        'blackman',
        'bohman'
    ])
plt.savefig(f'histograma_frecuencia_{SnraDb}DB.png')
plt.show()

cell_data = [
    ['boxcar', *values['boxcar'][2]],
    ['flattop', *values['flattop'][2]],
    ['blackman', *values['blackman'][2]],
    ['bohman', *values['bohman'][2]]
    ]
colum_labels = ['$window$', '$\mu_a$', '$S_a$', '$V_a$']

fig, ax = plt.subplots(figsize=(12,6))
ax.axis('off')
table = ax.table(cellText=cell_data, colLabels=colum_labels, loc='center' )
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 4)  # Escala ancho y alto
plt.savefig(f'tabla_valores_{SnraDb}DB.png')
plt.show()