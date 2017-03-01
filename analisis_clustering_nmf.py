from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from scipy.sparse import csr_matrix

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as sil_score

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

# Enfoque con NMF
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer

normalizer = Normalizer()

ans = []

for dim in range(2, 10):

    err = []
    for rand_state in range(100):

        nmf = NMF(n_components = dim, max_iter = 1000, init = 'random',\
              random_state = rand_state)

        x_red = nmf.fit_transform(x_tfidf.toarray())
        err.append(nmf.reconstruction_err_)
        if nmf.reconstruction_err_ == min(err):
            rand_state_aux = rand_state
        
    nmf = NMF(n_components = dim, max_iter = 1000, init = 'random',\
          random_state = rand_state_aux)

    x_red = nmf.fit_transform(x_tfidf.toarray())
    x_red = normalizer.fit_transform(x_red)

    sil = []
#    plt.axes([0.20, 0.20, 0.70, 0.70])
    for clusters in range(2, 10):
   
        try:
            km = KMeans(n_clusters = clusters, \
                    n_init = 200, n_jobs = 1).fit(x_red)
 
            sil.append(sil_score(x_red, km.labels_))
            ans.append([dim, clusters, sil_score(x_red, km.labels_)])
	
	    if sil_score(x_red, km.labels_) == max(sil):
                clusters_max = clusters
        except:
            pass

#    plt.plot(range(2, 31), sil, '.-', markersize = 15, \
#                         label = 'Dim = ' + str(dim))
"""
plt.grid('on')
plt.xlabel('Number of clusters (k)', size = 20)
plt.ylabel('Silhouette score', size = 20)
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.ylim([0.00, 1.00])
plt.legend(loc = 'best')
plt.savefig('NMF_silhouette_dims.eps')
plt.show()
"""

ans_sorted = sorted(ans, key = lambda x: x[2], reverse = True)
print ans_sorted[:5]
