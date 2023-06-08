import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import math
from nltk.stem import PorterStemmer
import nltk
import re
import numpy as np
from numpy.linalg import norm
import nltk
import json
stopwords=[]
ps = PorterStemmer()
vector={}
from nltk.stem.wordnet import WordNetLemmatizer
json_value_idf=open("/home/subinay/Documents/data/pairwise_similarity/tf_idf/idf_value_corpus.json")
Idf_value= json.load(json_value_idf)
dir_annotated="/home/subinay/Documents/data/pairwise_similarity/Second_50/50_docs/"
def top_terms(score,term):  # to find embedding of top tf-idf terms
    feature_scores = {}
    for i in range(len(score)):
        feature_scores[term[i]]=score[i]*Idf_value[term[i]]
    term_score=dict(sorted(feature_scores.items(), key=lambda item: item[1],reverse=True))
    docs1=[]
    for key,value in term_score.items():
        if(value!=0):
            docs1.append(key)
    num_to_select = int(0.1* len(docs1)) # to select top terms based on their tf-idf value
    # select the first 10% of the elements
    selected_elems = docs1[:num_to_select]
    tfidf_value={}
    for elem in selected_elems:
        tfidf_value[elem]=feature_scores[elem]
    return tfidf_value          
docs=[]
docid=[]
files=os.listdir(dir_annotated)
for file in files: 
    docid.append(file)
    text=""
    with open("/home/subinay/Documents/data/pairwise_similarity/Second_50/50_docs/"+file,"r") as f1:
        for line in f1:
            line=line.strip()
            text=text+line
    docs.append(text)
with open("/home/subinay/Documents/data/pairwise_similarity/stopword/stopword_10000_200.txt","r") as f1:
    for line in f1:
        line=line.strip()
        stopwords.append(line)
vectorizer = CountVectorizer()
vectorizer.set_params(stop_words=stopwords)
stemmer = PorterStemmer()
analyzer = vectorizer.build_analyzer()
vectorizer.set_params(tokenizer=lambda doc: (stemmer.stem(token) for token in analyzer(doc)))
X = vectorizer.fit_transform(docs)
# Access the vocabulary and feature names
vocabulary = vectorizer.vocabulary_
term= vectorizer.get_feature_names()
X1=[]
score=[]
count=0
for i in range(0,50):
    count=count+1
    x=X[i,].toarray()
    x1=x.flatten()
    score.append(x1)
for i in range(len(score)):
    text=top_terms(score[i],term)
    X1.append(text)
for i in range(len(docid)):
    vector[docid[i]]=X1[i]
with open("tf_idf_embedding_50docs.json", "w") as outfile:
     json.dump(vector, outfile)

















































































# for feature in feature_names:
#     print(feature)
# def tokenize(train_texts):
#   filtered_tokens = []
#   tokens = [word for sent in nltk.sent_tokenize(train_texts) for word in nltk.word_tokenize(sent)]
#   for token in tokens:
#     token=ps.stem(token)
#     if (('http' not in token) and ('@' not in token) and ('<.*?>' not in token)and (not token in stop_words)):
#             filtered_tokens.append(token)
#   return filtered_tokens
# def tokenize_stem(train_texts):
#   tokens = tokenize(train_texts)
#   stemmed_tokens = [lemmatizer.lemmatize(word) for word in tokens]

#   stemmer = SnowballStemmer('english')
#   stemmed_tokens = [stemmer.stem(token) for token in tokens]
#and (not token in stop_words))
  #return stemmed_tokens
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
#         stemmed_text=stemmed_text+" "+t
#     text_stemmed.append(stemmed_text)

    # for t in allwords_stemmed:
    #     stemmed_text=stemmed_text+" "+t.lower()
    # text_stemmed.append(stemmed_text)
#print(text_stemmed[0])
# X1=[] 
# # for i in range(0,1):
# #     print(text_stemmed[i])
# X = vectorizer.fit_transform(text_stemmed)
# term = vectorizer.get_feature_names_out()
# score=[]
# score2=[]
# count=0  # all terms in collection

