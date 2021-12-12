
from time import time
from datetime import date, timedelta

import streamlit as st
from streamlit import date_input#, expander, number_input, selectbox, warning, experimental_memo as st_cache

from sklearn.datasets import fetch_20newsgroups as news

from .scraper import scrape
from .misc import dwrite



@st.experimental_memo 
def retrieve(dataset):    
    if dataset == 'sklearn20news':
        return news(subset='all', remove=('headers','footers','quotes')).data
    
    
    
AVAIL_DATA = ['sklearn20news', 'wikipedia (developmental)', 'arxiv (coming next)']
DATA_MSG = 'The app is optimized for the Sklearn dataset.  \nOther options allow you to build a custom dataset for testing but tend to take a long time.  \n'
SCRAPE_MSG = "See what's trending on Wikipedia's Current Event portal.  \nEach day takes 3-5 minutes to scrape and increases model training time by roughly 1.2 times."

def get_data(last_n_days=1):
    dataset = st.selectbox('data source', AVAIL_DATA, index=0, help=DATA_MSG)

    if dataset == 'wikipedia (developmental)':
        default = [date.today()-timedelta(days=last_n_days), date.today()]
        dates = date_input('Get articles between:', default, date(2018,1,1), date.today(), help=SCRAPE_MSG)
        if len(dates)==1: return None

        t = time()
        art = scrape(*dates)
        scrape_time = (time()-t)//60

        dwrite(f'scrape {scrape_time} min\n')
        st.write(f'_Retrieved {len(art)} articles_')

        if len(art)<60: 
            warning('Corpus too small.  Try expanding the date range to get more documents.')
            return None
        elif len(art)>299:
            warning('Corpus might be too big.  Depending on RAM availability, app may become unstable due to high memory usage during training.') 
            
        dataset = {'name': 'wikipedia', 'data': art}
    elif dataset == 'sklearn20news':
        dataset = {'name': 'sklearn20news', 'data': retrieve(dataset)}
    else:
        return None
    
    return dataset




PASS_MSG = 'Number of passes through corpus, i.e., passes per mini-batch.  \nMore may improve model by facilitating convergence for small corpora at the cost of computation time.'
ITER_MSG = 'Number of E-step per document per pass.  \nHigher number may improve model by fascilitating document convergence at the cost of computation time.'

def get_param(t2v_nTopic):
    nTopic = int(st.number_input(
        'number of topics', 0, 999, 0, help=f'Larger number increases computation time.  \nBased on Top2Vec, we recommend {t2v_nTopic} for this dataset.'))
    with st.expander('optional training parameters'):
        passes = int(st.number_input('passes', 1, 99, 1, help=PASS_MSG))
        iters = int(st.number_input('iterations', 1, 999, 20, help=ITER_MSG))    
    return nTopic, passes, iters
