import igraph
import numpy as np

def threshold(weighted_matrix):

    for thr in np.linspace(0.00, 1.00, 41):

        adjacency_matrix = np.zeros(weighted_matrix.shape, dtype = np.int)

        for i in range(weighted_matrix.shape[0]):
            for j in range(weighted_matrix.shape[1]):
                if weighted_matrix[i][j] > thr:
                    adjacency_matrix[i][j] = 1

        graph_aux = igraph.Graph.Adjacency(list(adjacency_matrix), mode = igraph.ADJ_MAX)
        clust = graph_aux.clusters()
        giant = clust.giant()
        if len(giant.vs) < len(graph_aux.vs):
            thr = thr - 1.00/41
            break

    return thr
