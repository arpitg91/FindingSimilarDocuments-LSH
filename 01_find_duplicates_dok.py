'''
First basic script to get duplicate sentences.
The script just uses basic python to get the baseline time performance.
This will help guage the performance of numpy, scipy modules in the second script.
This script will take ages to run on the entire set of 10MM sentences.
The algorithm is O(n^2) in time and O(1) in memory.
'''
import sys, math
from gensim import corpora, models, similarities

raw_file = 'mmd_assignment/sentences.txt'
writer1 = open('mmd_assignment/sentences_similar_groups_dok.txt','wb')
writer2 = open('mmd_assignment/sentences_similar_sentences_dok.txt','wb')

texts = []
num_rows = int(sys.argv[1])
threshold = float(sys.argv[2])
counter = 0

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    #No need to below as the vectors are already normalised
    # denominator = math.sqrt(sum1) * math.sqrt(sum2)
    # if not denominator:
        # return 0.0
    # else:
        # return float(numerator/denominator)
    return float(numerator)

# Read the file and add to lists of sentences
for row in open(raw_file,'rb'):
    counter += 1
    if counter == num_rows:
        break
    texts.append(row.strip().split(' ')[1:])
    
# create dictionary of word:dummy_number
dictionary = corpora.Dictionary(texts)

# creating the Bag of Words for each documnet(row) in the data(only keep words in the dictionary)
corpus = [dictionary.doc2bow(text) for text in texts]
print "Bag of words"

#Normalise all the vectors in corpus
corp = []
for vec in corpus:
    dict_vec = dict(vec)
    l = dict_vec.values()
    sm = 0
    for sc in l:
        sm = sm + sc**2
    denom = math.sqrt(sm)
    for keys in dict_vec.keys():
        if denom != 0:
            dict_vec[keys] = dict_vec[keys]/denom
    corp.append(dict_vec)
    
# Get similarity of each document with other document
# For each similarity greater than threshold, make group of similar tickets
i = 0
group = {}
grp = 1
for vec1 in corp:
    i = i+1
    sim_max = 0
    index_max = -1
    index = i
    all = []
    for vec2 in corp[i:]:
        index = index + 1
        sim = get_cosine(vec1, vec2)
        if sim >= threshold:
            all.append(index)
            all.append(sim)
            if index in group.keys():
                group[i] = group[index]
            if i in group.keys():
                group[index] = group[i]
            if (i not in group.keys()) and (index not in group.keys()):
                grp += 1
                group[index] = grp
                group[i] = grp
    if len(all) > 0:
        writer1.write('|'.join(map(str,[i] + all))+'\n')
    if i%100 == 0 :
        print "Done with :%s rows" %i

# Write output file with calculated groups
similar_texts = set(group.keys())
counter = 0
for row in open(raw_file,'rb'):
    counter += 1
    if counter == num_rows:
        break
    if counter in similar_texts:
        writer2.write('|'.join(map(str,[group[counter]]+[counter] +[row])))

writer1.close()
writer2.close()
