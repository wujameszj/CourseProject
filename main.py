
from os import environ as env
from time import sleep
from datetime import date, timedelta

import nltk
import streamlit as st
from streamlit import sidebar, subheader, selectbox

from util.dataOp import get_data, get_param
from util.lda import MyLDA
from util.t2v import train_top2vec, relevant_topics_docs
from util.display import create_wordcloud, display_doc
from util.misc import filter_keywords, dwrite



MISC_MSG = ('_ _ _\n**Shoutout to Streamlit for generously hosting this app for free! \U0001f600**  \n- - -\n'
           f"App feels sluggish? Sorry about that.  \n_... Detecting ... {env.get('NUMBER_OF_PROCESSORS', 1)} worker available._  \n\n"
            'Run the app locally:  \n[Source code on Github](https://github.com/wujameszj/CourseProject)')


def main(debug):
    msg = st.empty()
    left, right = st.columns(2)
    left.header('Top2Vec'); right.header('LDA')
    
    
    with st.sidebar:
        st.subheader('Step 1: Choose dataset')    
        data = get_data()
        msg.info('Preparing Top2Vec'); sleep(3); msg.empty()  # give user time to correct input before start -- training cannot be stopped midway
    if not data: return   # invalid input; dont load rest of UI until new valid input is received 


    with left:
        t2v_model = train_top2vec(data)
       

    t2v_nTopic = t2v_model.get_num_topics()
    topics, _, __ = t2v_model.get_topics()


    with st.sidebar:
        st.subheader('Step 2: LDA parameters')
        lda_nTopic, passes, iters = get_param(t2v_nTopic)

        st.subheader('Step 3: Compare topics and documents')
        keyword = selectbox('search by keyword', filter_keywords(topics), help='This list consists of likely topic words in this dataset.')   # returns numpy_str

        st.write(MISC_MSG)
               
    
    DEFAULT_EXAMPLE = 6
    with left:
        if keyword:
            topicIDs, docIDs = relevant_topics_docs(t2v_model, keyword, DEFAULT_EXAMPLE)
            msg.info(f'Displaying top {DEFAULT_EXAMPLE} documents related to "{keyword}".')
        else:
            nWordcloud = min(DEFAULT_EXAMPLE, t2v_nTopic)
            topicIDs, docIDs = range(nWordcloud), range(DEFAULT_EXAMPLE)
            msg.info(f'Displaying {DEFAULT_EXAMPLE} topics and documents.')
            
        create_wordcloud(t2v_model, topicIDs)
        display_doc(data, docIDs)

    
    with right:
        if lda_nTopic:
            patient = st.info(f'Training model with {lda_nTopic} topics for {passes} passes and {iters} iterations. Please be patient.'); sleep(2)
            lda = MyLDA(data, lda_nTopic, passes, iters);  patient.empty()
            
            if keyword:
                nDoc = min(DEFAULT_EXAMPLE, lda_nTopic)
                topicIDs, docIDs = lda.relevant_topics_docs(keyword, nDoc)              
            else:
                nWordcloud = min(DEFAULT_EXAMPLE, lda_nTopic)
                topicIDs, docIDs = range(nWordcloud), range(DEFAULT_EXAMPLE)
            
            create_wordcloud(lda.model, topicIDs)
            display_doc(data, docIDs)        
#            if DEBUG: debug_msg.write(f'all topic words exist in LDA dict {all([True for word in topic_words if word in dictionary])}')



if __name__ == '__main__':
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')        
    
    st.set_page_config('CS410 Project', layout="wide")
    st.title('Compare Topic Modeling Algorithms')
    
    DEBUG = False
    if DEBUG:
        debug_msg = st.container()
    
    main(True)
    