
from top2vec import Top2Vec
from gensim.models import LdaModel 
import nltk
from sklearn.datasets import fetch_20newsgroups as news
import streamlit as st

from os import environ
from datetime import date, timedelta

from util.scraper import scrape
from util.lda import MyLDA
from util.display import create_wordcloud, display_doc



@st.experimental_memo  
def train_top2vec(data):
    if data['name'] == 'sklearn20news':
        return Top2Vec.load('models/20news.model')
    else:
        #return Top2Vec.load('models/20news.model')
        return Top2Vec(
            data['data'], min_count=9, keep_documents=False, 
            workers=int(environ.get('NUMBER_OF_PROCESSORS', 1))
        )

    
@st.experimental_memo  
def retrieve(dataset):    
    if dataset == 'sklearn20news':
        return news(subset='all', remove=('headers','footers','quotes')).data
    
   
AVAIL_DATA = ['sklearn20news', 'wikipedia', 'arxiv (in development)', 'reddit (in development)']

DATA_MSG = 'The app is optimized for the Sklearn dataset.  \nOther options allow you to build a custom dataset for testing but tend to take a long time.  \n'
SCRAPE_MSG = "See what's trending on Wikipedia's Current Event portal.  \nEach day takes 3-5 minutes to scrape and increases model training time by roughly 1.2 times."
PASS_MSG = 'Number of passes through corpus, i.e., passes per mini-batch.  \nMore may improve model by facilitating convergence for small corpora at the cost of computation time.'
ITER_MSG = 'Number of E-step per document per pass.  \nHigher number may improve model by fascilitating document convergence at the cost of computation time.'
MISC_MSG = ('_ _ _\n**Shoutout to Streamlit for generously hosting this app for free! \U0001f600**  \n- - -\n'
           f"App feels sluggish? Sorry about that.  \n_... Detecting ... {environ.get('NUMBER_OF_PROCESSORS', 1)} worker available._  \n\n"
            'Run the app locally:  \n[Source code on Github](https://github.com/wujameszj/CourseProject)')



def get_data(last_n_days=2):
    with st.sidebar:
        st.subheader('Step 1: Choose dataset')
        dataset = st.selectbox('data source', AVAIL_DATA, index=0, help=DATA_MSG)
        
        if dataset == 'wikipedia':
            default = [date.today()-timedelta(days=last_n_days), date.today()]
            dates = st.date_input('Get articles between:', default, date(2018,1,1), date.today(), help=SCRAPE_MSG)
            if len(dates)==1: return None
            
            art = scrape(*dates)
            st.write(f'_Retrieved {len(art)} articles_')
            data = {'name': 'wikipedia', 'data': art}
        elif dataset == 'sklearn20news':
            data = {'name': 'sklearn20news', 'data': retrieve(dataset)}
    return data


def get_param(t2v_nTopic):
    nTopic = int(st.number_input(
        'number of topics', 0, 999, 0, help=f'Larger number increases computation time.  \nBased on Top2Vec, we recommend {t2v_nTopic} for this dataset.'))
    with st.expander('optional training parameters'):
        passes = int(st.number_input('passes', 1, 99, 1, help=PASS_MSG))
        iters = int(st.number_input('iterations', 1, 999, 20, help=ITER_MSG))    
    return nTopic, passes, iters


@st.experimental_memo
def _filter(topics):
    return [None] + [words[0] for words in topics if len(words[0])>2] 


def main():
    msg = st.empty()
    left, right = st.columns(2)
    left.header('Top2Vec'); right.header('LDA')
    
    data = get_data()
    if not data: return   # invalid input; dont load rest of UI until new valid input is received 


    with left:
        t2v_model = train_top2vec(data)

    t2v_nTopic = t2v_model.get_num_topics()
    topics, _, __ = t2v_model.get_topics()


    with st.sidebar:
        st.subheader('Step 2: LDA parameters')
        lda_nTopic, passes, iters = get_param(t2v_nTopic)

        st.subheader('Step 3: Compare topics and documents')
        keyword = st.selectbox('search by keyword', _filter(topics), help='This list consists of likely topic words in this dataset.')   # returns numpy_str

        st.write(MISC_MSG)
               
    
    DEFAULT_EXAMPLE = 6
    with left:
        if keyword:
            nWordcloud, nDoc = 1, min(DEFAULT_EXAMPLE, t2v_nTopic)
            _,_,_, topicIDs = t2v_model.query_topics(str(keyword), nWordcloud)         # top2vec doesnt accept numpy_str, though LDA (gensim) does
            _, docIDs = t2v_model.search_documents_by_keywords([keyword], nDoc, return_documents=False, ef=len(data['data']))
            msg.info(f'Displaying top {nDoc} documents related to "{keyword}".')
        else:
            nWordcloud = min(DEFAULT_EXAMPLE, t2v_nTopic)
            topicIDs, docIDs = range(nWordcloud), range(DEFAULT_EXAMPLE)
            msg.info(f'Displaying {DEFAULT_EXAMPLE} topics and documents.')
            
        create_wordcloud(t2v_model, topicIDs)
        display_doc(data, docIDs)

    
    with right:
        if lda_nTopic:
            patient = st.info(f'Training model with {lda_nTopic} topics for {passes} passes and {iters} iterations. Please be patient.')
            lda = MyLDA(data, lda_nTopic, passes, iters);  patient.empty()
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
    
    DEBUG = False
    if DEBUG:
        debug_msg = st.container()
    
    main()
    