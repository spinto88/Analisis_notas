from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer
from sklearn.metrics import silhouette_score as sil_score

import codecs
import numpy as np
    
import igraph

from copy import deepcopy

from merit_factors import weak_merit_factor

 
directory = 'Notas_prueba/'
root_name = 'Notas_prueba'
max_notes = 31

max_df = 0.70
min_df = 5
nmf_dim_range = range(2, 20)

texts = []
for i in range(max_notes):

   try:
       fp = codecs.open(directory + root_name + str(i) + ".txt",'r','utf8')
       texts.append(fp.read())
       fp.close()
   except:
       pass

count_vect = CountVectorizer(ngram_range = (1,3), \
                             max_df = max_df, min_df = min_df, \
                             dtype = np.float64)

x_counts = count_vect.fit_transform(texts)

tfidf_transformer = TfidfTransformer(norm = 'l2')
x_tfidf = tfidf_transformer.fit_transform(x_counts)

# Weighted matrix of similarities
weighted_matrix = x_tfidf.dot(x_tfidf.T).toarray()

# Weighted matrix of dissimilarities
dissim = np.ones(weighted_matrix.shape) - weighted_matrix

# Complex weighted network

weighted_matrix_graph = deepcopy(weighted_matrix)
for i in range(len(weighted_matrix_graph)):
    weighted_matrix_graph[i][i] = 0.00

graph = igraph.Graph.Weighted_Adjacency(list(weighted_matrix_graph),\
                                        mode = igraph.ADJ_MAX)

weights = [es['weight'] for es in graph.es]

data = []

for dim in nmf_dim_range:

    observables_dim = []
  
    for rand_state in range(10):

        nmf = NMF(n_components = dim, max_iter = 1000, init = 'random',\
              random_state = rand_state)

        nmf_array = nmf.fit_transform(x_tfidf)

        # NMF labels
        labels = [np.argmax(x) for x in nmf_array]

        # ------ Silhouette coefficient of dissimilarities ---- #

        dissim = np.ones(weighted_matrix.shape) - weighted_matrix

        sil = sil_score(dissim, labels, metric = 'precomputed')

        # --------------------- Modularity -------------------- #

        mod = graph.modularity(labels, weights = weights)

        # ---------------- Weak merit factor ------------ #

        wmf = weak_merit_factor(graph, labels)


# ---------------------- Save the data ------------------ #

        observables_dim.append([sil, mod, wmf])

    data.append(observables_dim)


import cPickle as pk
pk.dump(data, open('Observables.pk', 'w'))

