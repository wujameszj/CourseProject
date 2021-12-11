
from top2vec import Top2Vec as T2V
from gensim.models import LdaModel 
import nltk
from sklearn.datasets import fetch_20newsgroups as news
import streamlit as st

from os import environ
from datetime import date, timedelta

from util.scraper import scrape
from util.lda import MyLDA
from util.display import create_wordcloud, display_doc
#from util.lda import calc_relevance, train_LDA
#from util.lda import *

from numpy.random import random


@st.experimental_memo  
def train_t2v(data):
    if data['name'] == 'sklearn20news':
        return T2V.load('models/20news.model')
    else:
        return T2V.load('models/20news.model')
        # return T2V(
        #     data['data'], min_count=9, keep_documents=False, 
        #     workers=int(environ.get('NUMBER_OF_PROCESSORS', 1))
        # )

    
@st.experimental_memo  
def retrieve(dataset):    
    if dataset == 'sklearn20news':
        return news(subset='all', remove=('headers','footers','quotes')).data
    
   
WORKER, CHUNKSIZE = environ.get('NUMBER_OF_PROCESSORS', 1), environ.get('CHUNK', 99999)
AVAIL_DATA = ['sklearn20news', 'wikipedia', 'arxiv (in development)', 'reddit (in development)']

SCRAPE_MSG = 'Scrape articles on Wikipedia\'s Current Event portal.  \n7-14 days tend to work well, not too few nor too many.'
PASS_MSG = 'Number of passes through corpus, i.e., passes per mini-batch.  \nMore may improve model by facilitating convergence for small corpora at the cost of computation time.'
ITER_MSG = 'Number of E-step per document per pass.  \nHigher number may improve model by fascilitating document convergence at the cost of computation time.'
MISC_MSG = ('_ _ _\n**Shoutout to Streamlit for generously hosting this app for free! \U0001f600**  \n- - -\n'
           f'App feels sluggish? Sorry about that.  \n_... Detecting ... {WORKER} worker available._  \n\n'
            'Run the app locally:  \n[Source code on Github](https://github.com/wujameszj/CourseProject)')



def get_data():
    with st.sidebar:
        st.subheader('Step 1: Get corpus')
        dataset = st.selectbox('data source', AVAIL_DATA, index=0, help='Choose dataset to perform topic modeling')
        
        if dataset == 'wikipedia':
            dates = []
#            while len(dates)!=2:
            dates = st.date_input('Get articles between:', [date.today()-timedelta(days=2), date.today()], date(2018,1,1), date.today(), help=SCRAPE_MSG)
            
            st.write(len(dates), dates)
            start, end = dates
            data = {'name': 'wikipedia', 'data': scrape(start, end)}
        elif dataset == 'sklearn20news':
            data = {'name': 'sklearn20news', 'data': retrieve(dataset)}
    return data


def get_param(nTopic):
    nTopic = int(st.number_input(
        'number of topics', 0, 999, 0, help=f'Larger number increases computation time.  \nBased on Top2Vec, we recommend {int(nTopic*.7)} for this dataset.'))
    with st.expander('optional training parameters'):
        passes = int(st.number_input('passes', 1, 99, 1, help=PASS_MSG))
        iters = int(st.number_input('iterations', 1, 999, 20, help=ITER_MSG))    
    return nTopic, passes, iters


def main():
    msg = st.empty()
    left, right = st.columns(2)
    left.header('Top2Vec'); right.header('LDA')
    
    data = get_data()
    t2v_model = train_t2v(data)
    
    nTopic = t2v_model.get_num_topics()
    topics, _, __ = t2v_model.get_topics(nTopic//2)
    
    DEFAULT_EXAMPLE = 3
    nExample = DEFAULT_EXAMPLE if DEFAULT_EXAMPLE < nTopic else nTopic 
    

    with st.sidebar:
        st.subheader('Step 2: LDA parameters')
        nTopic, passes, iters = get_param(nTopic)
            
        st.subheader('Step 3: Compare topics and documents')
        topic_words = [None] + [words[0] for words in topics if len(words[0])>2] 
        keyword = st.selectbox('search by keyword', topic_words, help='This list consists of likely topic words in this dataset.')   # returns numpy_str
        
        st.write(MISC_MSG)
               
    
    with left:
        if keyword:
            msg.info(f'Displaying top 6 documents related to "{keyword}".')
            _,_,_, topicIDs = t2v_model.query_topics(str(keyword), 1)         # top2vec doesnt accept numpy_str, though LDA (gensim) does
            _, docIDs = t2v_model.search_documents_by_keywords([keyword], nExample*2, return_documents=False, ef=len(data['data']))
        else:
            msg.info(f'Displaying {nExample*2} topics and documents.')
            topicIDs, docIDs = range(nExample*2), range(nExample*2)
            
        create_wordcloud(t2v_model, topicIDs)
        #display_doc(data, docIDs)

    
    with right:
        if nTopic:
            patient = st.info(f'Training model with {nTopic} topics for {passes} passes and {iters} iterations. Please be patient.')
            lda = MyLDA(data, nTopic, passes, iters);  patient.empty()
#            if DEBUG: debug_msg.write(f'all topic words exist in LDA dict {all([True for word in topic_words if word in dictionary])}')
           
            if keyword:
                topicIDs, docIDs = lda.relevant_topics_docs(keyword, nExample)              
            else:
                topicIDs, docIDs = range(nExample*2), range(nExample*2, nExample*4)
            
            create_wordcloud(lda.model, topicIDs)
            display_doc(data, docIDs)        



if __name__ == '__main__':
    #nltk.download('wordnet')
    #nltk.download('stopwords')
    
    st.set_page_config('CS410 Project', layout="wide")
    st.title('Compare Topic Modeling Algorithms')
    
    DEBUG = True
    if DEBUG:
        debug_msg = st.container()
    
    main()
    