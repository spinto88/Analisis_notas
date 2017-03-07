from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import homogeneity_score as hom_score

from sklearn.metrics import normalized_mutual_info_score as norm_score

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

from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer

mod = []
std_mod = []
score = []
labels_true = [0] * 10 + [1] * 7 + [2] * 7 + [3] * 7

dim_range = range(1, 32)

tfidf_transformer = TfidfTransformer(norm = 'l2')
x_tfidf = tfidf_transformer.fit_transform(x_counts)

for dim in dim_range:
  
    err = []
    mod2 = []
    score2 = []
    for rand_state in range(1000):

        nmf = NMF(n_components = dim, max_iter = 1000, init = 'random',\
              random_state = rand_state)

        x_red = nmf.fit_transform(x_tfidf.toarray())
        err.append(nmf.reconstruction_err_)
        if nmf.reconstruction_err_ == min(err):
            rand_state_aux = rand_state

        nmf = NMF(n_components = dim, max_iter = 1000, init = 'random',\
              random_state = rand_state_aux)

        nmf_array = nmf.fit_transform(x_tfidf)

        normalizer = Normalizer()
        nmf_array = normalizer.fit_transform(nmf_array)

        labels = [np.argmax(x) for x in nmf_array]
        score2.append(norm_score(labels_true, labels))

        weighted_matrix = nmf_array.dot(nmf_array.T)

        for i in range(len(weighted_matrix)):
            weighted_matrix[i][i] = 0.00

        import igraph
        graph = igraph.Graph.Weighted_Adjacency(list(weighted_matrix), mode = igraph.ADJ_MAX)
        weights = [es['weight'] for es in graph.es]
        mod2.append(graph.modularity(labels, weights = weights))

    mod.append(np.mean(mod2))
    std_mod.append(np.std(mod2))
    score.append(np.mean(score2))

np.save('Modularity.npy', mod)
np.save('Modularity_std.npy', std_mod)
np.save('Score.npy', score)
"""
plt.axes([0.15, 0.15, 0.70, 0.70])
plt.errorbar(dim_range, mod, std_mod, fmt = '.-', markersize = 20)
plt.grid('on')
plt.xlabel('Dimensions (D)', size = 20)
plt.ylabel('Modularity', size = 20)
plt.xlim([0, 32])
plt.ylim([0, 1.00])
plt.xticks(size = 20)
plt.yticks(size = 20)
#plt.savefig('Modularity_NMF.eps')
plt.show()
"""
"""
plt.clf()
plt.axes([0.15, 0.15, 0.70, 0.70])
plt.plot(dim_range, score, '.-', markersize = 20)
plt.grid('on')
plt.xlabel('Dimensions (D)', size = 20)
plt.ylabel('Homogeneity score', size = 20)
plt.xlim([0, 32])
plt.ylim([0.00, 1.05])
plt.xticks(size = 20)
plt.yticks(size = 20)
#plt.savefig('Number_communities_nmf.eps')
plt.show()
"""
"""
color_dict = {0: 'red', 1: 'gray', 2: 'green', 3: 'yellow'}


layout = graph.layout_fruchterman_reingold(weights=weights)
igraph.plot(graph, layout = layout, \
            edge_width = [5 * weight for weight in weights], \
            vertex_color = [color_dict[membership[i]] \
            for i in range(len(membership))], \
            target = 'Weighted_network_nmf4.png')
"""

