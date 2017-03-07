


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

import codecs
import numpy as np
import matplotlib.pyplot as plt
import random as rand

texts = []
for i in range(50):

    try:
       fp = codecs.open("Notas_prueba/Notas_prueba" + str(i) + ".txt",'r','utf8')
       texts.append(fp.read())
       fp.close()
    except:
        pass
 
count_vect = CountVectorizer(ngram_range = (1,3), \
                             max_df = 0.70, min_df = 2)
x_counts = count_vect.fit_transform(texts)
"""
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
"""
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer

mod = []

dim_range = [7] #range(1, 7)

tfidf_transformer = TfidfTransformer(norm = 'l2')
x_tfidf = tfidf_transformer.fit_transform(x_counts)

for dim in dim_range:
  
    err = []
    mod2 = []
    for rand_state in [90]:#range(1000):

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

    weighted_matrix = nmf_array.dot(nmf_array.T)

    for i in range(len(weighted_matrix)):
        weighted_matrix[i][i] = 0.00

    import igraph
    graph = igraph.Graph.Weighted_Adjacency(list(weighted_matrix), mode = igraph.ADJ_MAX)
    weights = [es['weight'] for es in graph.es]

    # Enfoque con umbral

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

    color_dict = {0: 'red', 1: 'gray', 2: 'green', 3: 'yellow', \
                  4: 'cyan', 5: 'orange', 6: 'violet', 7:'brown', 8:'white',\
                  9: 'black'}
    
    rand.seed(123457)
    layout = graph.layout_fruchterman_reingold(weights=weights)
    igraph.plot(graph, layout = layout, \
            edge_width = [3 * weight if weight >= thr else 0.00 for weight in weights], \
            vertex_color = [color_dict[labels[i]] \
            for i in range(len(labels))],
            vertex_label = [str(i) for i in range(len(graph.vs))],\
            vertex_size = 25)#,\
#            target = 'Layout_dim' + str(dim) + '.png')
    """
    components = [nmf.components_[i] \
              for i in range(len(nmf.components_))]

    features = count_vect.get_feature_names()
    fp = codecs.open('Interpretacion_nmf.txt','a','utf8')
    fp.write('Dimension ' + str(dim) + '\n\n')

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

