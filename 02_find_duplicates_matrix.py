'''
Second script for finding similar sentences.
The script reads documents and creates BOW matrix. This matrix is used to find similarity using arithematic operations.
The BOW matrix is a sparse matrix. However, similarity matrix is a dense matrix. So, the computation runs out of memory with higher BOW dimensions.
To accomodate this, the matrix is broken into parts and then arithematic operations are carried out.
The algorithm is O(n.lon(n)) in time and memory.
'''
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import binarize
import sys, math

raw_file = 'mmd_assignment/sentences.txt'
# writer = open('mmd_assignment/sentences_similar_sentences_matrix_%s.txt'%sys.argv[1],'wb')

texts = []
num_rows = int(sys.argv[1])
# threshold = float(sys.argv[2])
# num_rows = 10000
threshold = 0.95

# Read the file till num_rows. Find the num_rows where memory requirement increases beyond limit
counter = 0
for row in open(raw_file,'rb'):
    counter += 1
    if counter == num_rows:
        break
    texts.append(' '.join(row.strip().split(' ')[1:]))
    
# Make sparse numerical representation of the sentences
vectorizer = CountVectorizer(min_df=1)
X = normalize(vectorizer.fit_transform(texts).astype(np.float),norm='l2').tocsr()

# mag_squared,= np.array(X.multiply(X).sum(1).T)
# mag_all = np.sqrt(mag_squared) 
# mag_all[mag_all == 0] = 1
# del X,mag_squared,vectorizer

group = 0
group_map = {}

# Loop over documents taken num_documents2 at a time to find similarity. Taking all documents will lead to memory error
# Find similarity of documents and add them to the similar groups
num_documents2 = 2**12
for segment3 in range(0,int(math.ceil((1.0*len(texts))/num_documents2))):
    X1 = X[num_documents2*segment3:min(num_documents2*(segment3+1),len(texts))]
    X2 = X[0:min(num_documents2*(segment3+1),len(texts))]
    Y1 = binarize(X1 * X2.T, threshold=threshold).tocsr()
    # print Y1.shape
    del X1
    for row in range(Y1.shape[0]):
        sentence_index = row+num_documents2*segment3
        if row%1000 == 0:
            print "%d documents Done"%(row+segment3*num_documents2)
        cols_array = Y1[row].toarray()[0]
        cols = np.where(cols_array[0:row+segment3*num_documents2]>=threshold)[0]
        for col in reversed(cols):
            if sentence_index in group_map.keys():
                break
            try:
                value = Y1[row,col]
            except:
                value = 0
            if value >= threshold:
                if sentence_index != col:
                    if col in group_map.keys():
                        group_map[sentence_index] = group_map[col] 
                    else:
                        group += 1
                        group_map[sentence_index] = group
                        group_map[col]    = group

df1 = pd.DataFrame(pd.Series(group_map))
df1['Row'] = df1.index
df1.rename(columns={0:'Group'},inplace=True)
groups_map = df1.groupby('Group')['Row'].min().reset_index()
df2 = pd.merge(df1,groups_map,on='Group')
df2.index = df2['Row_x']
group = df2['Row_y'].to_dict()

similar_texts = set(group.keys())
counter = -1
for row in open(raw_file,'rb'):
    counter += 1
    if counter == num_rows:
        break
    if counter in similar_texts:
        writer.write('|'.join(map(str,[group[counter]]+[counter] +[row])))

writer.close()
