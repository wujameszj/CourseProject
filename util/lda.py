
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.corpus import stopwords

from gensim.models import LdaModel 
from gensim.corpora import Dictionary

import streamlit as st
from numpy import array, argmax, argpartition as argp, argsort

from os import environ


class MyLDA:
    
    def __init__(self, data, n, passes, iters):        
        self.nTopic = n
        
        self.corpus, self.dictionary = preprocess(data['data'])
        self.model = LdaModel(
            self.corpus, self.nTopic, self.dictionary.id2token, chunksize=environ.get('CHUNK', 99999), 
            passes=passes, iterations=iters, update_every=1, 
            alpha='auto', eta='auto', minimum_probability=0, eval_every=None
        )
        
        self.search_dict = list(self.dictionary.values())
        
        
    def relevant_topics_docs(self, word, nExample):
        topic_prob = self.model.get_term_topics(self.search_dict.index(word), minimum_probability=0)
        idx = argmax([p for i,p in topic_prob])
        topicIDs = [ topic_prob[idx][0] ]
        
        doc_prob = calc_relevance(self.corpus, self.search_dict.index(word))
        docIDs = argp(doc_prob, -nExample*2)[-nExample*2:]
        docIDs = docIDs[ argsort(doc_prob[docIDs])[::-1] ]    # list largest first  
                
        return topicIDs, docIDs
             
        
        
@st.experimental_memo
def calc_relevance(corpus, wordID):
    return array([
        sum([n if i==wordID else 0 for i,n in doc]) for doc in corpus
    ])
     

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




