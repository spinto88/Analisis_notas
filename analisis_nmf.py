from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from scipy.sparse import csr_matrix

import codecs
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score as sil_score


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

tfidf_transformer = TfidfTransformer(norm = None)
x_tfidf = tfidf_transformer.fit_transform(x_counts)

from sklearn.preprocessing import Normalizer
from sklearn.decomposition import NMF

norm = Normalizer()

for dim in range(2, 15):

    err = []

    for rand_state in range(100):

        nmf = NMF(n_components = dim, max_iter = 1000, init = 'random',\
              random_state = rand_state)

        x_red = nmf.fit_transform(x_tfidf.toarray())
        err.append(nmf.reconstruction_err_)
        if nmf.reconstruction_err_ == min(err):
            rand_state_aux = rand_state

    nmf = NMF(n_components = dim, max_iter = 1000, init = 'random',\
              random_state = rand_state)

    xred = nmf.fit_transform(norm.fit_transform(x_tfidf))

    labels = [np.argmax(x) for x in xred]

    print dim, sil_score(xred, labels)



