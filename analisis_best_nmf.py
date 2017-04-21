from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF

import codecs
import numpy as np
import matplotlib.pyplot as plt

"""
This programa shows the dependency of the dimensions
in the count_vectorizer method with the parameter max_df,
which filter a term if this one appears in more than the
(100 * max_df) percent of documents.
It helps to discard common terms, but if the corpus
is small and one topic is overrepresented then 
attention must be paid because important keywords
could be deleted.
"""

directory = 'Politica_4month/'
root_name = 'Nota'
max_notes = 2500

max_df = 0.70
min_df = 10

nmf_dim = 52

texts = []
for i in range(max_notes):

   try:
       fp = codecs.open(directory + root_name + str(i) + ".txt",'r','utf8')
       texts.append(fp.read())
       fp.close()
   except:
       pass


count_vect = CountVectorizer(ngram_range = (1,3), \
                          max_df = max_df, min_df = min_df)
x_counts = count_vect.fit_transform(texts)

tfidf_transformer = TfidfTransformer(norm = 'l2')
x_tfidf = tfidf_transformer.fit_transform(x_counts)

err = []
for rand_state in range(100):

    nmf = NMF(n_components = nmf_dim, max_iter = 1000, init = 'random',\
              random_state = rand_state)

    x_red = nmf.fit_transform(x_tfidf.toarray())
    err.append(nmf.reconstruction_err_)
    if nmf.reconstruction_err_ == min(err):
        rand_state_aux = rand_state

print 'Best seed for NMF: ', rand_state_aux
