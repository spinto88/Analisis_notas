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
plt.savefig('Dim_tfidf.eps')
plt.show()


