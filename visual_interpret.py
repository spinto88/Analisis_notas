from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer
from sklearn.metrics import silhouette_score as sil_score

import codecs
import numpy as np
import random as rand

    
import igraph

from copy import deepcopy

from merit_factors import weak_merit_factor

 
directory = 'Notas_prueba/'
root_name = 'Notas_prueba'
max_notes = 50

max_df = 20
min_df = 5

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

# Layout with threshold
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

 
weights = [es['weight'] for es in graph.es]

for n_components in range(5, 9):

    err = []
    for nmf_rand_state in rand.sample(range(1000),10):

        nmf = NMF(n_components = n_components, max_iter = 1000, init = 'random',\
              random_state = nmf_rand_state)

        nmf_array = nmf.fit_transform(x_tfidf)

        err.append(nmf.reconstruction_err_)
        if nmf.reconstruction_err_ == min(err):
            rand_state_aux = nmf_rand_state

    nmf = NMF(n_components = n_components, max_iter = 1000, init = 'random',\
              random_state = rand_state_aux)


    labels = [np.argmax(x) for x in nmf_array]
 
    rand.seed(123458)

    color_list = [tuple([(1 + rand.random())*0.5 for j in range(3)])\
              for i in range(len(labels))]

    layout = graph.layout_fruchterman_reingold(weights = weights)
    igraph.plot(graph, layout = layout, \
        edge_width = [5 * weight if weight >= thr else 0.00 \
                      for weight in weights], \
                      vertex_color = [color_list[labels[i]] \
                      for i in range(len(labels))],
                      vertex_label = [str(i) for i in range(len(graph.vs))],\
                      title = 'Comunities' + str(n_components),\
                      vertex_size = 25)

""" 
# Interpretation of components

components = [nmf.components_[i] \
        for i in range(len(nmf.components_))]

features = count_vect.get_feature_names()

fp = codecs.open('Interpretacion_nmf.txt','a','utf8')
fp.write('Dimension ' + str(nmf_dim) + '\n\n')

for j in range(len(components)):

    comp = components[j]
    feat_val = []
    for i in range(len(features)):
        feat_val.append([features[i], comp[i]])

    ans = sorted(feat_val, key = lambda x: x[1], reverse = True)
        
    notes_in_comp = [i for i in range(len(labels)) \
                         if labels[i] == j]

    for note in notes_in_comp:
        fp.write(str(note) + ', ')
    fp.write('\n')
    for t in ans[:20]:
        fp.write(t[0] + ', ')
    fp.write('\n\n')

fp.write('\n\n\n\n')
fp.close()
"""
