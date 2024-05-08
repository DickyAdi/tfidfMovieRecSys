import streamlit as st
from utils.utils import getAllMoviesTitle
# from script.scriptPredict import predict, lemmaToken
import script
import time
import pandas as pd
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any
import joblib

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
    
#load the trained tf-idf vectorizer
vectorizer = joblib.load('apps/models/tfidf.pkl')
#load the pre-processed pandas dataframe
master = pd.read_pickle('apps/data/processed/data.pkl')


def queryTokenize(title:str) -> Any:
    """Vector a query/title to be processed on the prediction function.

    Args:
        title (str): title of the movie that want to be vectorized

    Returns:
        Any: An sklearn tf-idf sparse matrix from title
    """
    query = master[master['original_title'] == title]['oneLiner'].squeeze()
    queryVec = vectorizer.transform([query])
    return queryVec

master = pd.read_pickle('apps/data/processed/data.pkl')

def predict(title:str) -> List:
    """rank 5 similar movies based on a given movie title.

    Args:
        title (str): title of the movie that want to be searched

    Returns:
        List: 5 most similar movies in order.
    """
    queryVec = queryTokenize(title)
    res = []
    matrix = np.array(master[['original_title', 'tfidf']])
    for i in range(len(matrix)):
        if matrix[i][0] == title:
            continue
        else:
            res.append([matrix[i][0], cosine_similarity(queryVec, matrix[i][1])[0][0]])
    sortedRes = sorted(res, key=lambda x: x[1], reverse=True)
    return sortedRes[:5]


allMovies = getAllMoviesTitle()
st.set_page_config(page_title='Recommendation page')
st.header('Movie recommendation system')
st.subheader('Choose one of your favourite movie')
textInput = st.selectbox('Select your movie', allMovies)
if st.button('Find!'):
    res = pd.DataFrame(predict(textInput), columns=['Title', 'Similarity score'])
    with st.spinner('Wait for recommendation...'):
        time.sleep(.5)
        st.table(res)