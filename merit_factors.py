import numpy as np

def weak_merit_factor(weighted_matrix, labels):

     wmf = []
     n_nodes = len(labels)

     for l in set(labels):
     
         communities_strength = []
         norm_l = np.sum([np.sum(weighted_matrix[i][:]) \
                   for i in range(n_nodes) \
                   if labels[i] == l])

         for i in range(n_nodes):
         
             if labels[i] == l:

                 ki = [weighted_matrix[i][j] \
		       if labels[j] == l \
                       else (-1.00) * weighted_matrix[i][j] \
                       for j in range(n_nodes)]

                 communities_strength.append(np.float(np.sum(ki))/norm_l)

         wmf.append(np.sum(communities_strength))

     return np.mean(wmf) 
