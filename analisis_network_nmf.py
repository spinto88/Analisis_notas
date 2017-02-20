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

weighted_matrix = x_tfidf.dot(x_tfidf.transpose()).toarray()

for i in range(len(weighted_matrix)):
    weighted_matrix[i][i] = 0.00

import igraph
graph = igraph.Graph.Weighted_Adjacency(list(weighted_matrix), mode = igraph.ADJ_MAX)
weights = [es['weight'] for es in graph.es]

clust = graph.community_infomap(edge_weights = weights)
membership = clust.membership

print graph.modularity(membership, weights = weights)

from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer

mod = []
dim_range = range(1, 32)
for dim in dim_range:

  try:
    nmf = NMF(dim)
    nmf_array = nmf.fit_transform(x_tfidf)

    normalizer = Normalizer()
    nmf_array = normalizer.fit_transform(nmf_array)

    weighted_matrix = nmf_array.dot(nmf_array.transpose())

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
    plt.ylim([-0.5, 5])
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.title('Bin size = 0.1', size = 20)
    plt.savefig('Weight_hist_nmf4.eps')
    plt.show()

    try:
      clust = graph.community_infomap(edge_weights = weights)
      membership = clust.membership
    except:
      pass

    mod.append(graph.modularity(membership, weights = weights))

  except:
    pass


color_dict = {0: 'red', 1: 'gray', 2: 'green', 3: 'yellow'}


layout = graph.layout_fruchterman_reingold(weights=weights)
igraph.plot(graph, layout = layout, \
            edge_width = [5 * weight for weight in weights], \
            vertex_color = [color_dict[membership[i]] \
            for i in range(len(membership))], \
            target = 'Weighted_network_nmf4.png')


plt.axes([0.15, 0.15, 0.70, 0.70])
plt.plot(dim_range, mod, '.-', markersize = 20)
plt.grid('on')
plt.xlabel('Dimensions (D)', size = 20)
plt.ylabel('Modularity', size = 20)
plt.xlim([0, 32])
plt.ylim([0, 1.00])
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.savefig('Modularity_NMF.eps')
plt.show()

"""
color_dict = {0: 'red', 1: 'gray', 2: 'green', 3: 'yellow'}


layout = graph.layout_fruchterman_reingold(weights=weights)
igraph.plot(graph, layout = layout, \
            edge_width = [5 * weight for weight in weights], \
            vertex_color = [color_dict[membership[i]] \
            for i in range(len(membership))], \
            target = 'Weighted_network_nmf4.png')
"""

