import re
import nltk
import json
from nltk.lm import MLE, Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.preprocessing import padded_everygrams
from sklearn.model_selection import train_test_split
import pickle

def preprocess(sentence):
    sentence=sentence.lower()
    sentence=re.sub(r"[^\w\s]","",sentence)
    return sentence.split()

def load_data(path):
    corpus=[]
    with open(path,"r",encoding="utf-8") as f:
        for data in f:
            line=json.loads(data)
            if line["label"]=="cluttering":
                continue
            tokens=preprocess(line["text"].strip())
            if tokens:
                corpus.append(tokens)
    return corpus

corpus=load_data("dataset.jsonl")
n=5
train_corpus,test_corpus=train_test_split(corpus,test_size=0.2,random_state=0)
train_data, vocab=padded_everygram_pipeline(n,train_corpus)

model=Laplace(n)
model.fit(train_data,vocab)

# test_sentence = preprocess("i love language modeling")

test_ngrams = []
for senetence in test_corpus:
    test_ngrams.extend(list(padded_everygrams(n,senetence)))
    
print("Perplexity:", model.perplexity(test_ngrams))

with open("5gram.pkl","wb") as f:
    pickle.dump(model,f)