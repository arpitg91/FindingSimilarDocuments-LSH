'''
Third script for finding similar sentences.
1. Read the file. Create hashing vectorizer. This is similar to BOW matrix. 
    a. The columns in BOW matrix is the word corpus
    b. The columns now are hash(word corpus)
This limits the number of columns in the matrix also introducing some error in the caculation.
2. Hashing matrix is used to find similarity matrix using arithematic operations.
3. The Hashing matrix is a sparse matrix. However, similarity matrix is a dense matrix. So, the computation runs out of memory with #documents.
4. To accomodate this, the matrix is broken into parts and then arithematic operations are carried out.
5. The algorithm is O(n.lon(n)) in time and O(n.lon(n)) in memory.
6. The accuracy is found by comparing against the groups created in second script.
'''
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import binarize
from sklearn.metrics.pairwise import cosine_similarity
import sys, math
import datetime, logging, cPickle

num_rows = 2**int(sys.argv[1])
hashing_ratio =float(sys.argv[2])
threshold = 0.95

# Initialise logging info with time stamp
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("Starting Process: python " + ' '.join(sys.argv))

raw_file = '03_similar_sentences/sentences.txt'
writer = open('03_similar_sentences/02_hashing_similar_sentences/%d_%d.txt'%(num_rows,int(hashing_ratio)),'wb')

# Read the file till num_rows. Find the num_rows where memory requirement increases beyond limit
texts = []
vocabulary = set()
counter = 0
for row in open(raw_file,'rb'):
    counter += 1
    if counter == num_rows:
        break
    vocabulary_sentence = row.strip().split(' ')[1:]
    texts.append(' '.join(vocabulary_sentence))
    vocabulary |= set(vocabulary_sentence)
    
hashing_size = int(len(vocabulary)/hashing_ratio)
# vectorizer = HashingVectorizer(norm=u'l2',n_features=hashing_size)
# X = vectorizer.fit_transform(texts).tocsr()

group = 0
group_map = {}

num_documents2 = 2**12
for segment3 in range(0,int(math.ceil((1.0*len(texts))/num_documents2))):
    texts1 = texts[num_documents2*segment3:min(num_documents2*(segment3+1),len(texts))]
    texts2 = texts[0:min(num_documents2*(segment3+1),len(texts))]
    bag_vocabulary = set([x for text in texts for x in text.split(' ')])
    
    vectorizer = HashingVectorizer(norm=u'l2',n_features=int(len(bag_vocabulary)/hashing_ratio))
    vectorizer.fit(texts)
    
    X1 = vectorizer.transform(texts1).astype(np.float).tocsr()
    X2 = vectorizer.transform(texts2).astype(np.float).tocsr()
    
    # X1 = X[num_documents2*segment3:min(num_documents2*(segment3+1),len(texts))]
    # X2 = X[0:min(num_documents2*(segment3+1),len(texts))]
    Y1 = binarize(X1 * X2.T, threshold=threshold).tocsr()
    # print Y1.shape
    del X1
    for row in range(Y1.shape[0]):
        sentence_index = row+num_documents2*segment3
        # if row%1000 == 0:
            # print "%d documents Done"%(row+segment3*num_documents2)
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
# print group, len(group_map)
# print group_map

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
logging.info("Completed Process: python %s. Finding Accuracy Now"%(' '.join(sys.argv)))

# Find the accuracy of hasher
actuals = pd.read_csv('03_similar_sentences/01_actual_similar_sentences/%d.txt'%num_rows,delimiter='|',header=None,names=['Group','SNum','Sentence'])
duplicates = pd.concat([group.sort(columns='SNum')[1:] for key,group in actuals.groupby('Group')])[['SNum']]
duplicates_hasher = pd.concat([group.sort(columns='Row_x')[1:] for key,group in df2.groupby('Group')])[['Row_x']]
all_identified = pd.merge(duplicates,duplicates_hasher,how='outer',left_on='SNum', right_on='Row_x').rename(columns={'SNum': 'FP','Row_x': 'FN'})
logging.info("%s Actual Duplicates: %d, FP: %d, FN: %d"%(' '.join(sys.argv),all_identified['FP'].notnull().sum(),all_identified['FP'].isnull().sum(),all_identified['FN'].isnull().sum()))
