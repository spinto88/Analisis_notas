import numpy as np
import matplotlib.pyplot as plt
import cPickle as pk

dim_range = range(2, 71)

dim_range_x = []

sil, sil_std, mod, mod_std, wmf, wmf_std = \
           [], [], [], [], [], []


for dim in dim_range:

    try:

        data_dim = pk.load(file('Observables.pk','r'))[dim_range.index(dim)]
    #    data_dim = pk.load(file('results_temporal6/Observables_dim' + str(dim) + '.pk','r'))

        sil.append(np.mean([x[0] for x in data_dim]))
        sil_std.append(np.std([x[0] for x in data_dim]))

        mod.append(np.mean([x[1] for x in data_dim]))
        mod_std.append(np.std([x[1] for x in data_dim]))

        wmf.append(np.mean([x[2] for x in data_dim]))
        wmf_std.append(np.std([x[2] for x in data_dim]))

        dim_range_x.append(dim)

    except:
        pass


plt.figure(1)
plt.axes([0.15, 0.15, 0.70, 0.70])
plt.errorbar(dim_range_x, sil, sil_std, fmt = '.-', markersize = 20)
plt.grid('on')
plt.xlabel('Dimensions (D)', size = 20)
plt.ylabel('Silhouette score', size = 20)
plt.ylim([0.00, 1.00])
#plt.xscale('symlog')
plt.xticks(size = 20)
plt.yticks(size = 20)
#plt.savefig('Silhouette_score_new_approach.eps')

plt.figure(2)
plt.axes([0.15, 0.15, 0.70, 0.70])
plt.errorbar(dim_range_x, mod, mod_std, fmt = '.-', markersize = 20)
plt.grid('on')
plt.xlabel('Dimensions (D)', size = 20)
plt.ylabel('Modularity', size = 20)
#plt.xscale('symlog')
plt.ylim([0.00, 1.00])
plt.xticks(size = 20)
plt.yticks(size = 20)
#plt.savefig('Modularity_new_approach.eps')

plt.figure(3)
plt.axes([0.15, 0.15, 0.70, 0.70])
plt.errorbar(dim_range_x, wmf, wmf_std, fmt = '.-', markersize = 20)
plt.grid('on')
plt.xlabel('Dimensions (D)', size = 20)
plt.ylabel('Weak merit factor', size = 20)
#plt.xscale('symlog')
plt.ylim([-1.00, 1.00])
plt.xticks(size = 20)
plt.yticks(size = 20)
#plt.savefig('Weak_merit_factor_new_approach.eps')

plt.show()
