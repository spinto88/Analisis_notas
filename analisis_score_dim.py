import numpy as np
import matplotlib.pyplot as plt

Sil = np.load('Sil.npy')

dim_range = range(1, 32)

score = np.load('Score.npy')

plt.axes([0.15, 0.15, 0.70, 0.70])
plt.plot(dim_range, score, '.-', markersize = 20)
plt.grid('on')
plt.xlabel('Dimensions (D)', size = 20)
plt.ylabel('Mutual information score', size = 20)
plt.xlim([0, 32])
plt.ylim([0.00, 1.05])
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.savefig('Score.eps')
plt.show()

