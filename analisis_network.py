from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from scipy.sparse import csr_matrix

import codecs
import numpy as np
import matplotlib.pyplot as plt

texts = []
for i in range(40):

   try:
       fp = codecs.open("Notas_prueba/Notas_prueba" + str(i) + ".txt",'r','utf8')
       texts.append(fp.read())
       fp.close()
   except:
       pass

 
count_vect = CountVectorizer(ngram_range = (1,3), \
                             max_df = 0.70, min_df = 2)
x_counts = count_vect.fit_transform(texts)

tfidf_transformer = TfidfTransformer(norm = 'l2')
x_tfidf = tfidf_transformer.fit_transform(x_counts)

weighted_matrix = (x_tfidf.dot(x_tfidf.transpose())).toarray()

for i in range(len(weighted_matrix)):
    weighted_matrix[i][i] = 0.00

import igraph
graph = igraph.Graph.Weighted_Adjacency(list(weighted_matrix), mode = igraph.ADJ_MAX)
weights = [es['weight'] for es in graph.es]

hist, edges = np.histogram(weights, bins = np.arange(-0.05, 1.15, 0.1), normed = True)

plt.axes([0.15, 0.15, 0.70, 0.70])
plt.plot([(edges[i] + edges[i+1])*0.5 for i in range(len(edges) - 1)], \
          hist, '.-', markersize = 20)
plt.grid('on')
plt.xlabel('Edge weights', size = 20)
plt.ylabel('Density', size = 20)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.5, 4.5])
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.title('Bin size = 0.1', size = 20)
plt.savefig('Weight_hist.eps')
plt.show()

"""
try:
    clust = graph.community_fastgreedy(weights = weights)
    print clust.as_clustering()
except:
    pass

try:
  clust = graph.community_label_propagation(weights = weights)
  print clust
except:
  pass

try:
  clust = graph.community_optimal_modularity(weights = weights)
  print clust
except:
  pass

try:
  clust = graph.community_edge_betweenness()
  print clust.as_clustering()
except:
  pass
"""
try:
  clust = graph.community_infomap(edge_weights = weights)
  print clust
except:
  pass

membership = clust.membership
print graph.modularity(membership, weights = weights)

print len(set(membership))

color_dict = {0: 'red', 1: 'gray', 2: 'green', 3: 'yellow'}


layout = graph.layout_fruchterman_reingold(weights=weights)
igraph.plot(graph, layout = layout, \
            edge_width = [5 * weight for weight in weights], \
            vertex_color = [color_dict[membership[i]] \
            for i in range(len(membership))], \
            target = 'Weighted_network.png')


# Enfoque con umbral

largest_size = []
for thr in np.linspace(1.00, 0.00, 21):

    adjacency_matrix = np.zeros(weighted_matrix.shape, dtype = np.int)

    for i in range(weighted_matrix.shape[0]):
        for j in range(weighted_matrix.shape[1]):
            if weighted_matrix[i][j] > thr:
                adjacency_matrix[i][j] = 1

    graph = igraph.Graph.Adjacency(list(adjacency_matrix), mode = igraph.ADJ_MAX)
    clust = graph.clusters()
    giant = clust.giant()
    largest_size.append(len(giant.vs))

plt.axes([0.15, 0.15, 0.70, 0.70])
plt.plot(np.linspace(1.00, 0.00, 21), largest_size, '.-', markersize = 20)
plt.grid('on')
plt.xlabel('Weight threshold', size = 20)
plt.ylabel('Size of largest connected component', size = 20)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.5, 32])
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.title('Binary network', size = 20)
plt.savefig('Largest component.eps')
    
adjacency_matrix = np.zeros(weighted_matrix.shape, dtype = np.int)

for i in range(weighted_matrix.shape[0]):
    for j in range(weighted_matrix.shape[1]):
        if weighted_matrix[i][j] > 0.10:
           adjacency_matrix[i][j] = 1

graph = igraph.Graph.Adjacency(list(adjacency_matrix), mode = igraph.ADJ_MAX)

clust = graph.community_infomap()
membership = clust.membership

print graph.modularity(membership)

color_dict = {0: 'red', 1: 'gray', 2: 'green', 3: 'yellow'}

layout = graph.layout_fruchterman_reingold()
igraph.plot(graph, layout = layout, \
            vertex_color = [color_dict[membership[i]] \
            for i in range(len(membership))],
            target = 'Binary_network.png')

