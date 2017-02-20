from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from scipy.sparse import csr_matrix

import codecs
import numpy as np
import matplotlib.pyplot as plt

texts = []

texts = ['La casa de Julia', 'La casa de Cristian', \
         'De la casa de Seba']

count_vect = CountVectorizer(ngram_range = (1,1), \
                             max_df = 1.00, min_df = 1)

x_counts = count_vect.fit_transform(texts)

tfidf_transformer = TfidfTransformer(norm = None)
x_tfidf = tfidf_transformer.fit_transform(x_counts)

print count_vect.get_feature_names()
