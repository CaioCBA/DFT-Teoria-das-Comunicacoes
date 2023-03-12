import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

n = 100

s = np.random.randn(1,n)
h = np.ones((64,1))
x = np.convolve(s.ravel(),h.ravel())
X = fft(x)

fig, axs = plt.subplots(3)
axs[0].plot(abs(X))
axs[0].set_title("Escala linear", fontsize=12)

axs[1].plot(20*np.log10(abs(X)))
axs[1].set_title("Escala logarítmica", fontsize=12)

axs[2].plot(20*np.log10(np.fft.fftshift(abs(X))))
axs[2].set_title("Escala logarítmica e simétrico", fontsize=12)

for ax in axs:
    ax.label_outer()
plt.show()