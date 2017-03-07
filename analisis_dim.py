from sklearn.feature_extraction.text import CountVectorizer

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


directory = 'Politica_1month/'
root_name = 'Nota'
max_notes = 600


texts = []
for i in range(max_notes):

   try:
       fp = codecs.open(directory + root_name + str(i) + ".txt",'r','utf8')
       texts.append(fp.read())
       fp.close()
   except:
       pass

 
dim = []
max_df_range = np.linspace(0.4, 1.00, 21)

for max_df in max_df_range:

    count_vect = CountVectorizer(ngram_range = (1,3), \
                                 max_df = max_df, min_df = 2)
    x_counts = count_vect.fit_transform(texts)

    dim.append(x_counts.shape[1])

plt.axes([0.20, 0.20, 0.70, 0.70])
plt.plot(max_df_range, dim, '.-', markersize = 15)
plt.xlabel('Max_df', size = 20)
plt.ylabel('Number of dimensions', size = 20)
plt.grid('on')
plt.xlim([0.35, 1.05])
plt.xticks(size = 20)
plt.yticks(size = 20)
#plt.savefig('Dim_tfidf.eps')
plt.show()


