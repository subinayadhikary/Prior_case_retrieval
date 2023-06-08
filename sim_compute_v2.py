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


































































# def cluster_eval():
#     k=len(ids_a)
#     for i in range(0,k):
#         for j in range(i+1,k):
#                 # list1=X[i,].toarray()
#                 # list1=list1.flatten()
#                 # list2=X[j,].toarray()
#                 # list2=list2.flatten()
#                 # A=X[i,].toarray()
#                 # A=A.ravel()
#                 # B=X[j,].toarray()
#                 # B=B.ravel()
#                 # z= np.dot(A,B)/(norm(A)*norm(B))
#                 #z=computeSim(X[i,].toarray(),X[j,].toarray())
#                 z=cosine_similarity(X1[i],X1[j])
#                 #z=cosine_similarity(list1,list2)
#                 #print(z)
#                 print(ids_a[i],ids_a[j],"   ",z)
# def cosine_similarity(v1,v2):
#     "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
#     sumxx, sumxy, sumyy = 0, 0, 0
#     for i in range(len(v1)):
#         x = v1[i]; y = v2[i]
#         sumxx += x*x
#         sumyy += y*y
#         sumxy += x*y
#     #print("sumxy",sumxy)
#     #print("sumxx",sumxx)
#     #print("sumyy",sumyy)
#     return sumxy/math.sqrt(sumxx*sumyy)




#top_docs=[]
# def top_terms(score,term):
#     #print(score)
#     #print(len(term))
#     feature_scores = {}
#     for i in range(len(score)):
#         #print(term[i],"##",score[i])
#         feature_scores[term[i]]=score[i]
#     term_score=dict(sorted(feature_scores.items(), key=lambda item: item[1],reverse=True))
#     docs1=[]
    
#     for key,value in term_score.items():
#             #docs1.append(key)
#         if(value!=0):
#             docs1.append(key)
        
#     num_to_select = int(0.1* len(docs1)) # to select top terms based on their tf-idf value
#     # select the first 20% of the elements
#     #print(num_to_select)
#     selected_elems = docs1[:num_to_select]
#     tfidf_value=[]
#     for i in range(len(score)):
#         tfidf_value.insert(i,0)
#     term=list(term)
#     print(selected_elems)
#     for elem in selected_elems:
#         index=term.index(elem)
#         value=feature_scores[elem]
#         tfidf_value[index]=value
#     #print(len(tfidf_value))

#     #print(len(selected_elems))
#     # print the selected elements
#     #text=" ".join(selected_elems)
#     return tfidf_value
# vectorizer = TfidfVectorizer()           
# docs=[]
# ids_a=[]
# files=os.listdir(dir_annotated)
# for file in files:
#     ids_a.append(file)
#     text=""
#     with open("/home/subinay/Documents/data/pairwise_similarity/processed_200_docs/"+file,"r") as f1:
#         for line in f1:
#             line=line.strip()
#             text=text+line
#     docs.append(text)
# with open("/home/subinay/Documents/data/pairwise_similarity/stopwords_10000_100.txt","r") as f1:
#     for line in f1:
#         line=line.strip()
#         stop_words.append(line)
# def tokenize(train_texts):
#   filtered_tokens = []
#   tokens = [word for sent in nltk.sent_tokenize(train_texts) for word in nltk.word_tokenize(sent)]
#   for token in tokens:
#     token=WordNetLemmatizer().lemmatize(token,'v')
#     if re.search('[a-zA-Z]',token):
#         if (('http' not in token) and ('@' not in token) and ('<.*?>' not in token)and (not token in stop_words)):
#             filtered_tokens.append(token)
#   return filtered_tokens
# j=0
# k=0
# all_content = []
# text_stemmed=[]
# vocab_tokenized = []
# vocab_stemmed = []
# for text in docs:
#     allwords_tokenized = tokenize(text)
#     #vocab_tokenized.append(allwords_tokenized)
#     # allwords_stemmed = tokenize_stem(text)
#     # vocab_stemmed.append(allwords_stemmed)
#     stemmed_text=""
#     for t in allwords_tokenized:
#         stemmed_text=stemmed_text+" "+t.lower()
#     text_stemmed.append(stemmed_text)

#     # for t in allwords_stemmed:
#     #     stemmed_text=stemmed_text+" "+t.lower()
#     # text_stemmed.append(stemmed_text)
# #print(text_stemmed[0])
# X1=[] 
# # for i in range(0,1):
# #     print(text_stemmed[i])
# X = vectorizer.fit_transform(text_stemmed)
# term = vectorizer.get_feature_names_out()
# score=[]
# score2=[]
# count=0  # all terms in collection
# for i in range(0,40):
#     count=count+1

#     x=X[i,].toarray()
#     x2=X[i,].toarray().tolist()
#     score2.append(x2)
#     x1=x.flatten()
#     #print(len(x1))
#     score.append(x1)
# for i in range(len(score)):
#     print(ids_a[i])
#     text=top_terms(score[i],term)
#     X1.append(text)
# # for i in range(len(X1[0])):
# #     print(X1[1][i],"##",score[1][i])

# #print(count)
# #print(docs[0])
# #print(term)
# #print(top_docs[0])
# #print(top_docs[1])
# #X1 =vectorizer.fit_transform(top_docs)
# #sum_sim=cluster_eval()
