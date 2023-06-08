import os
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from nltk.stem.snowball import SnowballStemmer
import nltk
import re
import numpy as np
from numpy.linalg import norm
import math
import json
stop_words=[]
json_embedding_tf_idf=open("/home/subinay/Documents/data/pairwise_similarity/tf_idf/tf_idf_embedding_50docs.json")
embedding= json.load(json_embedding_tf_idf)
dir_test="/home/subinay/Documents/data/pairwise_similarity/Second_50/50_docs/"
test_docs=os.listdir(dir_test)
sec_sim=[]
content_sim=[]
with open("/home/subinay/Documents/data/pairwise_similarity/Fourth_50/section_similarity_S_50.txt","r") as f2:
    for line in f2:
        line=line.strip()
        line=line.split()
        sec_sim.append(line[2])
def compute_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return round((numerator) / denominator,2)
    
def doc_sim(docid1,docid2): # compute similarity b/w two documents using cosine similarity
    score=0
    doc1_embedding=embedding[docid1]
    doc2_embedding=embedding[docid2]
    score=compute_cosine(doc1_embedding,doc2_embedding)
    return score 
def sim_test_docs():
    sim_score=0
    count=0
    for i in range(0,50):
        for j in range(i+1,50):
            sim_score=doc_sim(test_docs[i],test_docs[j])
            sim_score=1*float(sim_score)+0.0*float(sec_sim[count])
            count=count+1
            print(test_docs[i],test_docs[j],"   ",sim_score)
sim_test_docs()
