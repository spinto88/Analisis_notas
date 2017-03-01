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
for i in range(31):
 
    try:
        fp = codecs.open("Notas_prueba/Notas_prueba" + str(i) + ".txt",'r','utf8')
        texts.append(fp.read())
        fp.close()
    except:
        pass

 
count_vect = CountVectorizer(ngram_range = (1,3), \
                             max_df = 0.70, min_df = 2)

x_counts = count_vect.fit_transform(texts)

tfidf_transformer = TfidfTransformer(norm = None)
x_tfidf = tfidf_transformer.fit_transform(x_counts)

ans = []
for clusters in range(2, 40):

    try:
        km = KMeans(n_clusters = clusters, \
               n_init = 200, n_jobs = 1).fit(x_tfidf)

        ans.append([clusters, sil_score(x_tfidf, km.labels_)])
    except:
        pass

ans_sorted = sorted(ans, key = lambda x: x[1], reverse = True)
print ans_sorted[:5]
"""
# Informacion contenida en PCA
info = []
from sklearn.decomposition import PCA
#from sklearn.preprocessing import Normalizer
#normalizer = Normalizer()
for dim in range(1, 40):

    try:
        pca = PCA(n_components = dim)
        x_red = pca.fit_transform(x_tfidf.toarray())
        info.append(np.sum(pca.explained_variance_ratio_))
    except:
        pass

plt.plot(range(1,40), info, '.-', markersize = 20)
plt.xlabel('Dimensions', size = 20)
plt.ylabel('Explained variance ratio', size = 20)
plt.grid('on')
plt.ylim([0, 1])
plt.show()
#plt.savefig('Explained_variance.eps')
"""

# Enfoque con PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()

ans = []
ans2 = []

for dim in range(2, 21):

    pca = PCA(n_components = dim, whiten = True)
    x_red = pca.fit_transform(x_tfidf.toarray())
#    x_red = normalizer.fit_transform(x_red)

    for clusters in range(2, 31):
   
        try:
            km = KMeans(n_clusters = clusters, \
                    n_init = 200, n_jobs = 1).fit(x_red)

            ans.append([dim, clusters, sil_score(x_red, km.labels_)])
        except:
            pass

ans_sorted = sorted(ans, key = lambda x: x[2], reverse = True)
print ans_sorted[:5]

 
# Enfoque con PCA
from sklearn.decomposition import PCA

pca = PCA(n_components = 3, whiten = True)
x_red = pca.fit_transform(x_tfidf.toarray())

from sklearn.preprocessing import Normalizer

#normalizer = Normalizer()

#x_red = normalizer.fit_transform(x_red)

clusters = 4
km = KMeans(n_clusters = clusters, \
               n_init = 10, n_jobs = -1).fit(x_red)

print km.labels_

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(x_red)):

    if i < 10:
        color = 'black'
    elif i >= 10 and i < 17:
        color = 'blue'
    elif i >= 17 and i < 24:
        color = 'red'
    else:
        color = 'green'
    
    ax.scatter(x_red[i][0], x_red[i][1], x_red[i][2], c = color, marker =  'o')


ax.set_xlabel('Dim 1')
ax.set_ylabel('Dim 2')
ax.set_zlabel('Dim 3')
#plt.savefig('Kmeans_normalizer.eps')
plt.show()

