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

from sklearn.manifold import MDS
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer

mds = MDS(n_components = 2, dissimilarity = 'precomputed', random_state = 0)

norm = Normalizer()
err = []
dim = 6

for rand_state in range(100):

   nmf = NMF(n_components = dim, max_iter = 1000, init = 'random',\
         random_state = rand_state)

   x_red = nmf.fit_transform(x_tfidf.toarray())
   err.append(nmf.reconstruction_err_)
   if nmf.reconstruction_err_ == min(err):
       rand_state_aux = rand_state

nmf = NMF(n_components = dim, max_iter = 1000, init = 'random',\
          random_state = rand_state_aux)

x_red = nmf.fit_transform(x_tfidf)
x_red = norm.fit_transform(x_red)

labels = [np.argmax(x) for x in x_red]

components = norm.fit_transform(nmf.components_)

weighted_matrix = components.dot(components.T)
weighted_matrix_points = (x_red.dot(x_red.T))

dissim = 1.00 - weighted_matrix
dissim_points = 1.00 - weighted_matrix_points

mds = MDS(n_components = 2, dissimilarity = 'precomputed', random_state = 0)
print dissim.shape
print dissim_points.shape

dissim_red = mds.fit_transform(dissim)

mds = MDS(n_components = 2, dissimilarity = 'precomputed', random_state = 4)
dissim_red_points = mds.fit_transform(dissim_points)

color_dict = {0: 'red', 1: 'green', 2: 'blue', 3: 'orange', \
              4: 'violet', 5: 'brown'}

plt.axes([0.15, 0.15, 0.75, 0.75])
"""
for i in range(len(dissim_red_points)):
    plt.scatter(dissim_red_points[i,0], dissim_red_points[i,1], s = 100,
                c = color_dict[labels[i]])
"""
for i in range(len(dissim_red)):
    plt.arrow(0, 0, dissim_red[i,0], dissim_red[i,1], color = color_dict[i], width = 0.005)
             
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.xlabel('Dim 1', size = 20)
plt.ylabel('Dim 2', size = 20)
plt.grid('on')
plt.savefig('MDS_components' + str(dim) + '.eps')
plt.show()

