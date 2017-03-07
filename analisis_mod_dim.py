import numpy as np
import matplotlib.pyplot as plt


dim_range = range(1, 31)

mod = np.load('results2/Modularity.npy')
mod_std = np.load('results2/Modularity_std.npy')

plt.axes([0.15, 0.15, 0.70, 0.70])
plt.errorbar(dim_range, mod, mod_std, fmt = '.-', markersize = 20)
plt.grid('on')
plt.xlabel('Dimensions (D)', size = 20)
plt.ylabel('Modularity', size = 20)
plt.xlim([0, 32])
plt.ylim([0.00, 1.05])
plt.xticks(size = 20)
plt.yticks(size = 20)
#plt.savefig('Modularity_score.eps')
plt.show()

