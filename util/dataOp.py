
from datetime import date, timedelta

import streamlit as st

from sklearn.datasets import fetch_20newsgroups as news

from .scraper import scrape

        
        
@st.experimental_memo
def filter_keywords(topics, minWords=10):
    wordsPerTopic = 1 if len(topics) > minWords else minWords//len(topics)
    return [None] + [words[i] for words in topics for i in range(wordsPerTopic) if len(words[i])>2] 


    
AVAIL_DATA = ['sklearn20news', 'wikipedia', 'arxiv (coming next)']
DATA_MSG = 'The app is optimized for the Sklearn dataset.  \nOther options allow you to build a custom dataset for testing but tend to take a long time.'
SCRAPE_MSG = "See what's trending on Wikipedia's Current Event portal.  \nEach day takes 1-2 minutes to scrape and increases model training time by roughly 1.2 times."
BIG_WARN = 'Corpus is a bit big. Consider shorterning date range.  \nApp may become unstable due to high memory usage during training.'



@st.experimental_memo 
def get_news():    
    return news(subset='all', remove=('headers','footers','quotes')).data
    
    
    
def get_data(last_n_days=2):
    dataset = st.selectbox('data source', AVAIL_DATA, index=0, help=DATA_MSG)

    if dataset == AVAIL_DATA[1]:
        default = [date.today()-timedelta(days=last_n_days), date.today()]
        dates = st.date_input('Get articles between:', default, date(2018,1,1), date.today(), help=SCRAPE_MSG)
        if len(dates)==1: return None

        articles = scrape(*dates)
        st.write(f'_Retrieved {len(articles)} articles_')

        if len(articles)<60: 
            st.error('Corpus too small. Try expanding the date range by one day to get more documents.')
            return None
        elif len(articles)>399:
            st.warning(BIG_WARN) 
            
        dataset = {'name': AVAIL_DATA[1], 'data': articles}
    elif dataset == AVAIL_DATA[0]:
        dataset = {'name': AVAIL_DATA[0], 'data': get_news()}
    else:
        return None
    
    return dataset




PASS_MSG = 'Number of passes through corpus, i.e., passes per mini-batch.  \nMore may improve model by facilitating convergence for small corpora at the cost of computation time.'
ITER_MSG = 'Number of E-step per document per pass.  \nHigher number may improve model by fascilitating document convergence at the cost of computation time.'

def get_param(t2v_nTopic):
    nTopic = int(st.number_input(
        'number of topics', 0, 999, 0, help=f'Larger number increases computation time.  \nA suggested number for this dataset based on Top2Vec is {t2v_nTopic}.'))
    with st.expander('optional training parameters'):
        passes = int(st.number_input('passes', 1, 99, 1, help=PASS_MSG))
        iters = int(st.number_input('iterations', 1, 999, 20, help=ITER_MSG))    
    return nTopic, passes, iters


