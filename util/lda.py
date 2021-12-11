
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.corpus import stopwords

from gensim.models import LdaModel 
from gensim.corpora import Dictionary

import streamlit as st
from numpy import array

from os import environ



@st.experimental_memo #(suppress_st_warning=True)
def preprocess(data, below=2, above=.5):
    regex, lemma = RegexpTokenizer(r'\w+'), WordNetLemmatizer()
    en_stop = set(stopwords.words('english'))
    useful = lambda token: True if token not in en_stop and len(token) > 2 and not token.isnumeric() else False
    
    docs = [regex.tokenize(doc.lower()) for doc in data]
    docs = [[token for token in doc if useful(token)] for doc in docs]
    #docs = [[lemma.lemmatize(token) for token in doc] for doc in docs]

    dictionary = Dictionary(docs)
    before = len(dictionary)
    dictionary.filter_extremes(no_below=below, no_above=above, keep_n=None)  # keep all
#    if DEBUG: debug_msg.write(f'filter_extremes removed {before} -> {len(dictionary)}')
    
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    _ = dictionary[0]  # This is only to "load" the dictionary.
    return corpus, dictionary
    
    
@st.experimental_memo 
def train_LDA(data, nTopic, passes, iters):
    corpus, dictionary = preprocess(data['data'])
    model = LdaModel(
        corpus, nTopic, dictionary.id2token, chunksize=environ.get('CHUNK', 99999), passes=passes, iterations=iters, update_every=1, 
        alpha='auto', eta='auto', minimum_probability=0, eval_every=None
    ) 
    return model, list(dictionary.values()), corpus


@st.experimental_memo
def calc_relevance(corpus, wordID):
    return array([
        sum( [n if i==wordID else 0 for i,n in doc] ) for doc in corpus
    ])

