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

dissim = 1.00 - weighted_matrix

from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as sil_score


for dim in range(2, 5):

  mds = MDS(n_components = dim, dissimilarity = 'precomputed')

  dissim_red = mds.fit_transform(dissim)

  sil = []
  for k in range(2, 31):
    try:
        km = KMeans(k).fit(dissim_red)
        sil.append(sil_score(dissim_red, km.labels_))
    except:
        pass

  plt.axes([0.20, 0.20, 0.70, 0.70])
  plt.plot(range(2, 31), sil, '.-', markersize = 15, \
                         label = 'Dim = ' + str(dim))

plt.grid('on')
plt.xlabel('Number of clusters (k)', size = 20)
plt.ylabel('Silhouette score', size = 20)
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.legend(loc = 'best')
plt.savefig('MDS.eps')
plt.show()


mds = MDS(n_components = 2, dissimilarity = 'precomputed')
dissim_red = mds.fit_transform(dissim)

km = KMeans(4).fit(dissim_red)
labels = km.labels_

color_dict = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow'}

plt.axes([0.15, 0.15, 0.75, 0.75])
for i in range(len(dissim_red)):
    plt.scatter(dissim_red[i,0], dissim_red[i,1], s = 100, \
                c = color_dict[labels[i]])

plt.xlabel('Dim 1', size = 20)
plt.ylabel('Dim 2', size = 20)
plt.grid('on')
plt.savefig('MDS_map.eps')
plt.show()

