from sklearn.feature_extraction.text import CountVectorizer

import codecs
import numpy as np
 
directory = 'Politica_4month/'
root_name = 'Nota'
max_notes = 2500

max_df = 0.70
min_df = 10

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

total_components = x_counts.shape[0] * x_counts.shape[1]
non_zero_components = x_counts.count_nonzero()

estimated_number_of_clusters = np.float(total_components)/non_zero_components

print 'Number of documents: ' + str(x_counts.shape[0])
print 'Number of features: ' + str(x_counts.shape[1])
print 'Estimated number of clusters: ' + str(estimated_number_of_clusters)


