
from top2vec import Top2Vec
import streamlit as st

from os import environ
from time import time

from .displayIO import dwrite


N_PROC = int(environ.get('NUMBER_OF_PROCESSORS', 1))


@st.experimental_memo(suppress_st_warning=True)
def train_top2vec(data, compromise=True):
    
    if data['name'] == 'sklearn20news':
        return Top2Vec.load('models/20news.model')
    else:
        corpus = data['data']
        nDoc = len(corpus)
        min_doc_freq = max(5, int(nDoc/99))
        
        if compromise:
            speed = 'learn' if nDoc < 99 else 'fast-learn'
        else:
            speed = 'learn'
        
        t = time(); patient = st.info(f'Please be patient... _speed={speed}, min_doc_freq={min_doc_freq}_')
        
        model = Top2Vec(
            corpus, min_count=min_doc_freq, keep_documents=False, speed=speed,
            workers=N_PROC if N_PROC < 2 else N_PROC-1
        )
        
        dwrite(f't2v {(time()-t)//60} min\n');  patient.empty()
        
        return model
    
    
    
def relevant_topics_docs(t2v_model, keyword, nDoc):
    _,_,_, topicIDs = t2v_model.query_topics(str(keyword), num_topics=1)         # top2vec doesnt accept numpy_str, though LDA (gensim) does
    _, docIDs = t2v_model.search_documents_by_keywords([keyword], nDoc, return_documents=False, ef=99999)  # len(data['data']))
    return topicIDs, docIDs