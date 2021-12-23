
from os import environ
from time import time

from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.corpus import stopwords

from gensim.models import LdaModel 
from gensim.corpora import Dictionary

import streamlit as st
from numpy import array, argmax, argpartition as argp, argsort

from .displayIO import dwrite



class MyLDA:
    
    def __init__(self, data, n, passes, iters):        
        patient = st.info(f'Training model with {n} topics for {passes} passes and {iters} iterations. Please be patient.')
        
        self.nTopic = n
        self.corpus, self.dictionary = preprocess(data['data'])
        self.vocab = list(self.dictionary.values())

        t = time()
        self.model = train(self.corpus, self.dictionary.id2token, n, passes, iters)
        dwrite(f'LDA {(time()-t)//60} min\n');  patient.empty()
        
        
    def relevant_topics_docs(self, word, nDoc):
        topic_prob = self.model.get_term_topics(self.vocab.index(word), minimum_probability=0)
        idx = argmax([p for i,p in topic_prob])
        topicIDs = [ topic_prob[idx][0] ]
        
        doc_prob = calc_relevance(self.corpus, self.vocab.index(word))
        docIDs = argp(doc_prob, -nDoc)[-nDoc:]
        docIDs = docIDs[ argsort(doc_prob[docIDs])[::-1] ]    # list largest first  
                
        return topicIDs, docIDs
             
        
        
@st.experimental_memo 
def calc_relevance(corpus, wordID):
    return array([
        sum([n if i==wordID else 0 for i,n in doc]) for doc in corpus
    ])
     

    
@st.experimental_memo 
def preprocess(data, above=.5):
    regex, lemma = RegexpTokenizer(r'\w+'), WordNetLemmatizer()
    en_stop = set(stopwords.words('english'))
    useful = lambda token: True if token not in en_stop and len(token) > 2 and not token.isnumeric() else False
    
    docs = [regex.tokenize(doc.lower()) for doc in data]
    docs = [[token for token in doc if useful(token)] for doc in docs]
    #docs = [[lemma.lemmatize(token) for token in doc] for doc in docs]

    min_doc_freq = max(9, len(data)//999)
    dictionary = Dictionary(docs)

    before = len(dictionary)    
    dictionary.filter_extremes(no_below=min_doc_freq, no_above=above, keep_n=None)  # keep all
    dwrite(f'filter_extremes {before}->{len(dictionary)}')
    
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    _ = dictionary[0]  # This is only to "load" the dictionary.
    
    return corpus, dictionary
    
    
    
@st.experimental_memo 
def train(corpus, id2token, nTopic, passes, iters):
    return LdaModel(
        corpus, nTopic, id2token, chunksize=int(environ.get('CHUNK', 99999)), 
        passes=passes, iterations=iters, update_every=1, 
        alpha='auto', eta='auto', minimum_probability=0, eval_every=None
    ) 
