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
import matplotlib.pyplot as plt

 
directory = 'Politica_1month/'
root_name = 'Nota'
max_notes = 600

max_df = 50
texts = []
for i in range(max_notes):

   try:
       fp = codecs.open(directory + root_name + str(i) + ".txt",'r','utf8')
       texts.append(fp.read())
       fp.close()
   except:
       pass

count_vect = CountVectorizer(ngram_range = (1,3), \
                             max_df = max_df, min_df = 10, \
                             dtype = np.float64)

x_counts = count_vect.fit_transform(texts)

tfidf_transformer = TfidfTransformer(norm = 'l2')
x_tfidf = tfidf_transformer.fit_transform(x_counts)

print x_tfidf.shape

# Weighted matrix of similarities
weighted_matrix = x_tfidf.dot(x_tfidf.T).toarray()

# Complex weighted network

weighted_matrix_graph = deepcopy(weighted_matrix)
for i in range(len(weighted_matrix_graph)):
    weighted_matrix_graph[i][i] = 0.00

graph = igraph.Graph.Weighted_Adjacency(list(weighted_matrix_graph),\
                                        mode = igraph.ADJ_MAX)

weights = [es['weight'] for es in graph.es]

hist = np.histogram(weights, bins = np.arange(-0.05, 1.10, 0.10), normed = True)[0]
plt.plot(np.arange(0.00, 1.05, 0.1), hist, '.-', markersize = 20)
plt.yscale('log')
plt.grid('on')
plt.show()
