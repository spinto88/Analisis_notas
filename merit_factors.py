import numpy as np


"""
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

     return np.sum(wmf)
"""
def weak_merit_factor(graph, labels):

    strength = []

    for l in set(labels):

        edges_in_com = [(i,j) for i in range(len(graph.vs))\
                              for j in range(len(graph.vs))\
                              if i < j and labels[i] == l \
                                       and labels[j] == l]

        edges_out_com = [(i,j) for i in range(len(graph.vs))\
                              for j in range(len(graph.vs))\
                              if i < j and ((labels[i] != l \
                                       and labels[j] == l) or \
                                       (labels[i] == l and labels[j] != l))]

        weights_in = np.sum([graph.es[edges]['weight'] for edges in edges_in_com])
        weights_out = np.sum([graph.es[edges]['weight'] for edges in edges_out_com])

        strength_com = (weights_in - weights_out)/(weights_in + weights_out)

        strength.append(strength_com)

    return np.mean(strength)

def strong_merit_factor(graph, labels):

    strength = []

    

    for l in set(labels):

        edges_in_com = [(i,j) for i in range(len(graph.vs))\
                              for j in range(len(graph.vs))\
                              if i < j and labels[i] == l \
                                       and labels[j] == l]

        edges_out_com = [(i,j) for i in range(len(graph.vs))\
                              for j in range(len(graph.vs))\
                              if i < j and ((labels[i] != l \
                                       and labels[j] == l) or \
                                       (labels[i] == l and labels[j] != l))]

        weights_in = np.sum([graph.es[edges]['weight'] for edges in edges_in_com])
        weights_out = np.sum([graph.es[edges]['weight'] for edges in edges_out_com])

        strength_com = (weights_in - weights_out)/(weights_in + weights_out)

        strength.append(strength_com)

    return np.mean(strength)
