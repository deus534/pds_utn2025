import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write


fs_ecg = 1000
mat_struct = sio.loadmat('ECG_TP4.mat')
patter1 = mat_struct['heartbeat_pattern1'].flatten()
patter2 = mat_struct['heartbeat_pattern2'].flatten()


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(patter1)

plt.subplot(1,2,2)
plt.plot(patter2)

plt.show()