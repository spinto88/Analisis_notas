from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer

import codecs
import numpy as np
import matplotlib.pyplot as plt

import datetime
from datetime import date
from datetime import timedelta
 
directory = 'Politica_4month/'
root_name = 'Nota'
max_notes = 2500

max_df = 0.70
min_df = 10

nmf_dim = 52
nmf_rand_state = 0

initial_date = date(2016, 8, 05)
final_date = date(2016, 12, 05)

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
                             max_df = max_df, min_df = min_df)
x_counts = count_vect.fit_transform(texts)

tfidf_transformer = TfidfTransformer(norm = 'l2')
x_tfidf = tfidf_transformer.fit_transform(x_counts)

nmf = NMF(n_components = nmf_dim, max_iter = 1000, init = 'random',\
              random_state = nmf_rand_state)

nmf_array = nmf.fit_transform(x_tfidf)

norm = Normalizer('l2')
nmf_array = norm.fit_transform(nmf_array)

# NMF labels
labels = [np.argmax(x) for x in nmf_array]
    
# Interpretation of components
components = [nmf.components_[i] \
        for i in range(len(nmf.components_))]

features = count_vect.get_feature_names()

fp = codecs.open('Interpretacion_nmf.txt','a','utf8')
fp.write('Dimension ' + str(nmf_dim) + '\n\n')

for j in range(len(components)):

    comp = components[j]
    feat_val = []
    for i in range(len(features)):
        feat_val.append([features[i], comp[i]])

    ans = sorted(feat_val, key = lambda x: x[1], reverse = True)
        
    notes_in_comp = [i for i in range(len(labels)) \
                         if labels[i] == j]

    for note in notes_in_comp:
        fp.write(str(note) + ', ')
    fp.write('\n')
    for t in ans[:20]:
        fp.write(t[0] + ', ')
    fp.write('\n\n')

fp.write('\n\n\n\n')
fp.close()

for j in range(len(components)):

    notes_in_comp = range(len(texts))
    notes_in_comp = [i for i in range(len(labels)) \
                         if labels[i] == j]

    notes_per_day = np.zeros((final_date - initial_date).days + 1, \
                    dtype = np.float)

    for i in notes_in_comp:
        notes_per_day[(dates[i]-initial_date).days] += (nmf_array[i][j]**2) * len(texts[i])

    plt.figure(1)
    plt.clf()
    plt.axes([0.20, 0.20, 0.70, 0.70])
    plt.plot(range(len(notes_per_day)), notes_per_day, '.-', markersize = 15)
    plt.grid('on')
    plt.xticks(range(0, len(notes_per_day), 10), \
              [initial_date + timedelta(i) for i in range(0, len(notes_per_day),10)], rotation = 'vertical')
#    plt.ylim([0, 20000])
    plt.savefig('Topic_' + str(j) + '.eps')

    """

    wm = 7 # Window mean
    topic_mean = [np.mean(notes_per_day[i:i+wm]) for i in range(len(notes_per_day) - wm - 1)]
    x_axis = [range(i, i + wm)[wm/2] for i in range(len(notes_per_day) - wm - 1)]

    plt.figure(2)
    plt.clf()
    plt.axes([0.20, 0.20, 0.70, 0.70])
    plt.plot(x_axis, topic_mean, '-')
    plt.grid('on')
#    plt.xticks(range(0, len(notes_per_day), 10), \
#              [initial_date + timedelta(i) for i in range(0, len(notes_per_day),10)], rotation = 'vertical')
#    plt.ylim([0, 20000])
    plt.savefig('Mean_Topic_' + str(j) + '.eps')
    """
