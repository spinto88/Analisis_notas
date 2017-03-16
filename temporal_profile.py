from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer
from sklearn.metrics import silhouette_score as sil_score

import codecs
import numpy as np
import random as rand

    
import igraph
import matplotlib.pyplot as plt

from copy import deepcopy

from merit_factors import weak_merit_factor
import datetime
from datetime import date

 
directory = 'Politica_1month/'
root_name = 'Nota'
max_notes = 600

initial_date = date(2016, 8, 5)
final_date = date(2016, 9, 5)

max_df = 50
nmf_dim = 20
nmf_rand_state = 7

texts = []
dates = []
for i in range(max_notes):

   try:
       fp = codecs.open(directory + root_name + str(i) + ".txt",'r','utf8')
       data = fp.readlines()
       dates.append(data[0])
       texts.append(''.join(data[1:]))
       fp.close()
   except:
       pass

dates = [datetime.datetime.strptime(date_string, '%Y-%m-%d\n').date() \
         for date_string in dates]

count_vect = CountVectorizer(ngram_range = (1,3), \
                             max_df = max_df, min_df = 2)
x_counts = count_vect.fit_transform(texts)

tfidf_transformer = TfidfTransformer(norm = 'l2')
x_tfidf = tfidf_transformer.fit_transform(x_counts)

normalizer = Normalizer(norm = 'l1')

nmf = NMF(n_components = nmf_dim, max_iter = 1000, init = 'random',\
              random_state = nmf_rand_state)

nmf_array = nmf.fit_transform(x_tfidf)


nmf_array = (normalizer.fit_transform(nmf_array.T)).T

# NMF labels
labels = [np.argmax(x) for x in nmf_array]

# Interpretation of components

components = [nmf.components_[i] \
        for i in range(len(nmf.components_))]


for j in range(len(components)):
 
    notes_in_comp = range(len(texts))
#    notes_in_comp = [i for i in range(len(labels)) \
#                         if labels[i] == j]

    notes_per_day = np.zeros((final_date - initial_date).days + 1, \
                    dtype = np.float)
    for i in notes_in_comp:
        notes_per_day[(dates[i]-initial_date).days] += nmf_array[i][j] * len(texts[i])

    plt.plot(range(len(notes_per_day)), notes_per_day, '.-')
    plt.grid('on')
    plt.show()

