
import nltk
import streamlit as st

from os import environ
from time import time
from datetime import date, timedelta

from util.dataOp import get_data
from util.lda import MyLDA
from util.t2v import train_top2vec
from util.display import create_wordcloud, display_doc
from util.misc import filter_keywords, dwrite


N_PROC = int(environ.get('NUMBER_OF_PROCESSORS', 1))
PASS_MSG = 'Number of passes through corpus, i.e., passes per mini-batch.  \nMore may improve model by facilitating convergence for small corpora at the cost of computation time.'
ITER_MSG = 'Number of E-step per document per pass.  \nHigher number may improve model by fascilitating document convergence at the cost of computation time.'
MISC_MSG = ('_ _ _\n**Shoutout to Streamlit for generously hosting this app for free! \U0001f600**  \n- - -\n'
           f"App feels sluggish? Sorry about that.  \n_... Detecting ... {N_PROC} worker available._  \n\n"
            'Run the app locally:  \n[Source code on Github](https://github.com/wujameszj/CourseProject)')


def get_param(t2v_nTopic):
    nTopic = int(st.number_input(
        'number of topics', 0, 999, 0, help=f'Larger number increases computation time.  \nBased on Top2Vec, we recommend {t2v_nTopic} for this dataset.'))
    with st.expander('optional training parameters'):
        passes = int(st.number_input('passes', 1, 99, 1, help=PASS_MSG))
        iters = int(st.number_input('iterations', 1, 999, 20, help=ITER_MSG))    
    return nTopic, passes, iters


def main(debug):
    msg = st.empty()
    left, right = st.columns(2)
    left.header('Top2Vec'); right.header('LDA')
    
    data = get_data()
    if not data: return   # invalid input; dont load rest of UI until new valid input is received 


    with left:
        t = time()
        t2v_model = train_top2vec(data)
        t2v_time = (time()-t)//60
        if debug: dwrite(f't2v {t2v_time} min\n')

    t2v_nTopic = t2v_model.get_num_topics()
    topics, _, __ = t2v_model.get_topics()


    with st.sidebar:
        st.subheader('Step 2: LDA parameters')
        lda_nTopic, passes, iters = get_param(t2v_nTopic)

        st.subheader('Step 3: Compare topics and documents')
        keyword = st.selectbox('search by keyword', filter_keywords(topics), help='This list consists of likely topic words in this dataset.')   # returns numpy_str

        st.write(MISC_MSG)
               
    
    DEFAULT_EXAMPLE = 6
    with left:
        if keyword:
            topicIDs, docIDs = relevant_topics_docs(t2v_model, keyword)
            msg.info(f'Displaying top {DEFAULT_EXAMPLE} documents related to "{keyword}".')
        else:
            nWordcloud = min(DEFAULT_EXAMPLE, t2v_nTopic)
            topicIDs, docIDs = range(nWordcloud), range(DEFAULT_EXAMPLE)
            msg.info(f'Displaying {DEFAULT_EXAMPLE} topics and documents.')
            
        create_wordcloud(t2v_model, topicIDs)
        display_doc(data, docIDs)

    
    with right:
        if lda_nTopic:
            patient = st.info(f'Training model with {lda_nTopic} topics for {passes} passes and {iters} iterations. Please be patient.')
            t = time()
            
            lda = MyLDA(data, lda_nTopic, passes, iters);  patient.empty()
            lda_time = (time()-t)//60
            if debug: dwrite(f'lda {lda_time} min\n')

#            if DEBUG: debug_msg.write(f'all topic words exist in LDA dict {all([True for word in topic_words if word in dictionary])}')
           
            if keyword:
                nDoc = min(DEFAULT_EXAMPLE, lda_nTopic)
                topicIDs, docIDs = lda.relevant_topics_docs(keyword, nDoc)              
            else:
                nWordcloud = min(DEFAULT_EXAMPLE, lda_nTopic)
                topicIDs, docIDs = range(nWordcloud), range(DEFAULT_EXAMPLE)
            
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
    
    main(True)
    