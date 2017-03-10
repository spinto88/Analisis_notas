import numpy as np
import matplotlib.pyplot as plt
import cPickle as pk

dim_range = range(2, 13)


sil, sil_std, mod, mod_std, wmf, wmf_std = \
           [], [], [], [], [], []


for dim in dim_range:

    data_dim = pk.load(file('results2/Observables2_dim' + str(dim) + '.pk','r'))

    sil.append(np.mean([x[0] for x in data_dim]))
    sil_std.append(np.std([x[0] for x in data_dim]))

    mod.append(np.mean([x[1] for x in data_dim]))
    mod_std.append(np.std([x[1] for x in data_dim]))

    wmf.append(np.mean([x[2] for x in data_dim]))
    wmf_std.append(np.std([x[2] for x in data_dim]))

plt.figure(1)
plt.axes([0.15, 0.15, 0.70, 0.70])
plt.errorbar(dim_range, sil, sil_std, fmt = '.-', markersize = 20)
plt.grid('on')
plt.xlabel('Dimensions (D)', size = 20)
plt.ylabel('Silhouette score', size = 20)
plt.ylim([0.00, 1.05])
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.savefig('Silhouette_score_2.eps')

plt.figure(2)
plt.axes([0.15, 0.15, 0.70, 0.70])
plt.errorbar(dim_range, mod, mod_std, fmt = '.-', markersize = 20)
plt.grid('on')
plt.xlabel('Dimensions (D)', size = 20)
plt.ylabel('Modularity', size = 20)
plt.ylim([0.00, 1.05])
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.savefig('Modularity_2.eps')

plt.figure(3)
plt.axes([0.15, 0.15, 0.70, 0.70])
plt.errorbar(dim_range, wmf, wmf_std, fmt = '.-', markersize = 20)
plt.grid('on')
plt.xlabel('Dimensions (D)', size = 20)
plt.ylabel('Weak merit factor', size = 20)
plt.ylim([0.00, 1.05])
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.savefig('Weak_merit_factor_2.eps')

plt.show()
