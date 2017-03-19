# Finding Similar Documents

The scripts in this repository have been used to find similar sentences using cosine distance. 
To find similarity, all sentences have to be compared against each other. This comparison means that the operations will be O(n^2) in time or memory.

However, with optimised calculations, the time-memory requirements can be controlled. Below are the four methologies used for similarity calculation.

## Dictionary of keys
- This method finds similarity of documents one by one thus populating the similarity matrix and finding and groups of similar sentences.
- Time complexity: O(n^2). Memory Complexity: O(1)
- Exact calculations
- This method is very good for small number of documents because of limited memory requirements.

## Matrix Multiplications
- Make BOW matrix and use matrix multiplication to find the similarity matrix
- BOW matrix is sparse matrix. However, similarity matrix is dense matrix and requires memory to reside.
- Exact calculations
- Very good for ~100K documents as it give exact calculations on a decent hardware.

## Hashing Vectorizer
- Make approximate BOW matrix using hashing of dictionary to find similarity matrix
- BOW matrix is sparse matrix. However, similarity matrix is dense matrix and requires memory to reside.
- Approximate calculations
- Very good for ~100k documents with long sentences and high vocabulary size. The vocabulary is artifically controlled by hashing.

## Locality Sensitive Hashing
- Bucket potentially similar documents in same subset using approximate linear algorithms.
- Similarity calculated only for a fraction of documents and not all the documents.
- Approximate calculations. The algorithm needs to be tuned for high accuracy.
- Very good for millions of documents. It can theoretically go upto infinite documents with proper tuning.
- Requires linear time and memory. For further details on the algorithm, lookup Mining Massive Datasets on Coursera-Stanford University