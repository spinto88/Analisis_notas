from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from scipy.sparse import csr_matrix

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import codecs
import numpy as np

texts = []
for i in range(71):

    try:
        fp = codecs.open("Politica/Nota" + str(i) + ".txt",'r','utf8')
        texts.append(fp.read())
        fp.close()
    except:
        pass


stop_words = stopwords.words('spanish')

count_vect = CountVectorizer(texts, ngram_range = (1,1),  \
                             max_df = 0.75, min_df = 3)

x_counts = count_vect.fit_transform(texts)
features_names = count_vect.get_feature_names()

tfidf_transformer = TfidfTransformer(norm = None)
tfidf_transformer.fit(x_counts)

x_tfidf = tfidf_transformer.transform(x_counts)

fp = codecs.open('Features.txt','a','utf8')

for text in range(len(texts)):

    names_tfidf = sorted([[features_names[i], x_tfidf[text, i]] \
                for i in range(len(features_names))], \
                key = lambda x: x[1], reverse = True)

    for name in names_tfidf[:20]:
        fp.write(name[0] + ' ' + str(name[1]) + '\n')
    fp.write('\n')

fp.close()

exit()

weighted_matrix = (x_tfidf.dot(x_tfidf.transpose())).toarray()

for i in range(len(weighted_matrix)):
    weighted_matrix[i][i] = 0.00

# Enfoque tipo red compleja

x_tfidf_array = x_tfidf.toarray()
weighted_matrix = np.zeros([len(texts), len(texts)], dtype = np.float)

import scipy.spatial.distance as distance

for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        weighted_matrix[i][j] = x_tfidf_array[i].dot(x_tfidf_array[j])

weighted_matrix += np.transpose(weighted_matrix)
 

import igraph
graph = igraph.Graph.Weighted_Adjacency(list(weighted_matrix), mode = igraph.ADJ_MAX)
weights = [es['weight'] for es in graph.es]

plt.hist(weights, bins = 100, normed = True)
plt.show()

layout = graph.layout_fruchterman_reingold(weights=weights)
igraph.plot(graph, layout = layout)



clust = graph.community_fastgreedy(weights = weights)
print clust.as_clustering()



membership = graph.community_infomap(edge_weights = weights)
print membership



"""
# Enfoque con PCA
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
x_dim2 = pca.fit_transform(x_tfidf.toarray())


import matplotlib.pyplot as plt

for i in range(len(x_dim2)):

    if i < 10:
        color = 'black'
    elif i >= 10 and i < 17:
        color = 'blue'
    elif i >= 17 and i < 24:
        color = 'red'
    else:
        color = 'green'
    
    plt.plot(x_dim2[i][0], x_dim2[i][1], '.', color = color, markersize = 20)
plt.grid('on')
plt.show()
"""








