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

nmf = NMF(n_components = 4)
x_red = nmf.fit_transform(x_tfidf.toarray())

x_red = normalizer.fit_transform(x_red)

components = [nmf.components_[i] \
              for i in range(len(nmf.components_))]

features = count_vect.get_feature_names()

for comp in components:

    feat_val = []
    for i in range(len(features)):
        feat_val.append([features[i], comp[i]])

    ans = sorted(feat_val, key = lambda x: x[1], reverse = True)

    print ans[:10]


