'''
1. Final script to find duplicates in large number of sentences. This script implements locality sensitive hashing. 
2. Cosine distance is implemented in this script as that is most popular for text documents. For other distances, different algorithms will have to be coded.
3. First the parameters are varied and LSH algorithm is run on a sample of documents.
4. The results are compared with actual simililarities to obtain parameters with maximum efficiency and fitting in available time/memory constraints.
5. The script is then run on all the documents.
6. The algorithm is O(n) in time and memory.
7. For details on the algorithm, you can look at the Mining Massive Datasets course (Coursera-Stanford University)
'''
import datetime, logging, cPickle, sys, math, os, time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import binarize
from scipy.sparse.csgraph import connected_components
from scipy import stats

num_rows = 2**int(sys.argv[1])
planes_band = int(sys.argv[2])
# nplanes = int(sys.argv[3])
buckets = 512
approximation = int(sys.argv[4])
# approximation = 1
# bands = int(nplanes/planes_band)
bands = int(sys.argv[3])
nplanes = bands*planes_band
threshold = 0.95

# Initialise logging info with time stamp
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("Starting Process: python " + ' '.join(sys.argv))
start_time = time.time()

raw_file = '03_similar_sentences/sentences.txt'
directory = '03_similar_sentences/03_lsh_similar_sentences/reducers/%s'%('_'.join(sys.argv[1:]))

if not os.path.exists(directory):
    os.makedirs(directory)

texts = []
counter = 0
for row in open(raw_file,'rb'):
    counter += 1
    if counter == num_rows:
        break
    vocabulary_sentence = row.strip().split(' ')[1:]
    texts.append(' '.join(vocabulary_sentence))
    
writers = [open(directory+'/%03d.TXT'%i,'wb') for i in range(buckets)]
# writer1 = open('03_similar_sentences/03_lsh_similar_sentences/%s.txt'%('_'.join(sys.argv)),'wb')

logging.info("Making BOW matrix")
vectorizer = CountVectorizer(min_df=1)
X = normalize(vectorizer.fit_transform(texts).astype(np.float),norm='l2').tocsr()

logging.info("Evaluating Hash Functions")
ref_planes = normalize(np.random.randn(nplanes, X.shape[1]),norm='l2')

logging.info("Encoding Bands")
X1 = binarize(X*ref_planes.T,threshold = 0)
encoder = np.matrix(2**np.arange(planes_band).reshape((planes_band,1)))

for band in range(bands):
    logging.info("Finding Similarity in Band %d"%band)
    X_band = np.array((X1[:,band*planes_band:band*planes_band+planes_band]*encoder).astype(np.int64)).reshape(X1.shape[0])
    X_band_collission = np.where(np.bincount(X_band)>1)[0]
    X_band_bucket = np.mod(X_band,buckets)
    logging.info("Writing candidates for Band %d"%band)
    for hashed_sentence in np.where(np.in1d(X_band, X_band_collission))[0]:
        writers[X_band_bucket[hashed_sentence]].write('%d|'%hashed_sentence+'%d|'%band+'%d|'%X_band[hashed_sentence]+texts[hashed_sentence]+'\n')

[writer.close() for writer in writers]

logging.info("All candidates found. Removing FP")

def get_similar_sentences(df):
    vectorizer = CountVectorizer(min_df=1)
    if approximation == 0:
        vectorizer = CountVectorizer(min_df=1)
        # try:
        X = normalize(vectorizer.fit_transform(df['SENTENCE']).astype(np.float),norm='l2').tocsr()
        Y = binarize(X * X.T, threshold=threshold).tocsr()
        df['COMPONENT'] = connected_components(Y)[1] 
        # except:
            # df['COMPONENT'] = 1
            # print df.head()
    else:
        df['COMPONENT'] = 1
    df_component = df.groupby('COMPONENT')['S.NUM'].agg({'COUNT' : np.size, 'GROUP' : min})
    df_component = df_component[df_component['COUNT']>=2]
    return pd.merge(df,df_component,how='inner',left_on = 'COMPONENT',right_index=True)

candidates = []
for bucket in range(buckets):
    df1 = pd.read_csv(directory+'/%03d.TXT'%bucket,header=None, names=['S.NUM','BAND','HASH', 'SENTENCE'],delimiter='|')
    # print len(df1)
    if len(df1) > 0:
        candidates.append(df1.groupby(['BAND','HASH']).apply(get_similar_sentences))
logging.info("Combining All results")
all_sentences = pd.concat(candidates)
all_neighbours = pd.DataFrame(all_sentences.groupby('S.NUM')['GROUP'].agg(lambda x: ' '.join(np.unique(x).astype(np.str))))
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(all_neighbours['GROUP'])
all_neighbours['FINAL GROUP'] = connected_components(X*X.T)[1] 
all_neighbours['SENTENCE_ID'] = all_neighbours.index
all_neighbours.to_csv('03_similar_sentences/03_lsh_similar_sentences/%s.txt'%('_'.join(sys.argv[1:])),index=False)

end_time = time.time()
logging.info("Finding Accuracy")
# Find the accuracy of hasher
actuals = pd.read_csv('03_similar_sentences/01_actual_similar_sentences/%d.txt'%num_rows,delimiter='|',header=None,names=['Group','S.NUM','Sentence'])
duplicates = pd.concat([group.sort(columns='S.NUM')[1:] for key,group in actuals.groupby('Group')])[['S.NUM']]
duplicates_hasher = pd.concat([group.sort(columns='SENTENCE_ID')[1:] for key,group in all_neighbours.groupby('FINAL GROUP')])[['SENTENCE_ID']]
all_identified = pd.merge(duplicates,duplicates_hasher,how='outer',left_on='S.NUM', right_on='SENTENCE_ID').rename(columns={'S.NUM': 'FP','SENTENCE_ID': 'FN'})
logging.info(' '.join(sys.argv[1:]) + " Time Taken: %0.0f seconds, Actual Duplicates: %d, FP: %d, FN: %d"%((end_time - start_time),all_identified['FP'].notnull().sum(),all_identified['FP'].isnull().sum(),all_identified['FN'].isnull().sum()))


#group = 1
#for bucket in range(buckets):
#    candidates = {}
#    # logging.info("Reading File %d"%bucket)
#    for row in open(directory+'/cosine_similarity_lsh/sentences_similar_sentences_lsh_ALL_CANDIDATES_%d_%d_%d_%d.TXT'%(num_rows,planes_band,bands,bucket),'rb'):
#        row_split = row.strip().split('|')
#        key = '|'.join(row_split[1:3])
#        value = ' '.join([row_split[0],row_split[3]])
#        candidates[key] = candidates.get(key,[]) + [value]
#    # logging.info("Candidates Loaded. Finding exact Similarity Now")
#    for key, value in candidates.iteritems():
#        X = normalize(vectorizer.fit_transform(value).astype(np.float),norm='l2').tocsr()
#        Y = binarize(X * X.T, threshold=threshold).tocsr()
#        component = connected_components(Y)[1] 
#        # group += component.max() + 1
#        freq = stats.itemfreq(component)
#        indices =  np.where(np.in1d(component,freq[freq[:,1]>1,0]))[0]
#        for text in np.asarray(value)[indices]:
#            writer1.write('|'.join([key,text])+'\n')
#writer1.close()

