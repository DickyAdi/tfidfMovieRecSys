import joblib
import os
import nltk
import pandas as pd
import numpy as np
from typing import Any
from nltk import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity


vectorizer = joblib.load('apps/models/tfidf.pkl')
master = pd.read_pickle('apps/data/processed/data.pkl')

class lemmaToken():
    def __init__(self) -> None:
        self.lemma = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
    def getPos(self, tag:str):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    def __call__(self, doc) -> Any:
        tokens = word_tokenize(doc)
        wnt = nltk.pos_tag(tokens)
        return [self.lemma.lemmatize(word, pos=self.getPos(tag)) for word, tag in wnt if word.lower() not in self.stopwords]

def queryTokenize(title:str):
    query = master[master['original_title'] == title]['oneLiner'].squeeze()
    queryVec = vectorizer.transform([query])
    return queryVec

def predict(title:str):
    queryVec = queryTokenize(title)
    res = []
    matrix = np.array(master[['original_title', 'tfidf']])
    for i in range(len(matrix)):
        if matrix[i][0] == title:
            continue
        else:
            res.append([matrix[i][0], cosine_similarity(queryVec, matrix[i][1])[0][0]])
    res.sort(key=lambda x: x[1], reverse=True)
    return res[:5]
