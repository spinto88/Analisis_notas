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
for i in range(71):
 
    try:
        fp = codecs.open("Notas_prueba/Notas_prueba" + str(i) + ".txt",'r','utf8')
        texts.append(fp.read())
        fp.close()
    except:
        pass


stop_words = stopwords.words('spanish')
                       
count_vect = CountVectorizer(ngram_range = (1,3), \
                             max_df = 0.70, min_df = 3)

x_counts = count_vect.fit_transform(texts)

tfidf_transformer = TfidfTransformer(norm = 'l2')
x_tfidf = tfidf_transformer.fit_transform(x_counts)

ans = []
for clusters in range(2, 10):
     km = KMeans(n_clusters = clusters).fit(x_tfidf)
     ans.append([clusters, sil_score(x_tfidf, km.labels_)])

ans_sorted = sorted(ans, key = lambda x: x[1], reverse = True)
print ans_sorted[:20]


# Enfoque con PCA
from sklearn.decomposition import PCA

ans = []

for dim in [3]: #range(2, 10):

    pca = PCA(n_components = dim)
    x_red = pca.fit_transform(x_tfidf.toarray())

    for clusters in [4]: #range(2, 10):

        km = KMeans(n_clusters = clusters, \
                    n_init = 100, n_jobs = -1).fit(x_red)

#        ans.append([dim, clusters, sil_score(x_red, km.labels_)])


#ans_sorted = sorted(ans, key = lambda x: x[2], reverse = True)
#print ans_sorted[:20]

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


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.savefig('Kmeans.png')
plt.show()


