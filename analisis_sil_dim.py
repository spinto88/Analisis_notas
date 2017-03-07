import numpy as np
import matplotlib.pyplot as plt

Sil = np.load('results2/Sil.npy')

dim_range = range(2, 31)

sil = [np.mean(x) for x in Sil]
sil_std = [np.std(x) for x in Sil]

plt.axes([0.15, 0.15, 0.70, 0.70])
plt.errorbar(dim_range, sil, sil_std, fmt = '.-', markersize = 20)
plt.grid('on')
plt.xlabel('Dimensions (D)', size = 20)
plt.ylabel('Silhouette score', size = 20)
#plt.xlim([0, 32])
plt.ylim([0.00, 1.05])
plt.xticks(size = 20)
plt.yticks(size = 20)
#plt.savefig('Silhouette_score.eps')
plt.show()

