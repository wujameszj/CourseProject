
from os import environ as env
from time import sleep
from datetime import date, timedelta

import nltk
import streamlit as st
from streamlit import sidebar, subheader, selectbox

from util.dataOp import get_data, get_param, filter_keywords
from util.lda import MyLDA
from util.t2v import train_top2vec, relevant_topics_docs
from util.displayIO import create_wordcloud, display_doc, dwrite



AUTH_MSG = ('_ _ _\n**Shoutout to Streamlit for generously hosting this app for free! \U0001f600**  \n- - -\n'
           f"App feels sluggish? Sorry about that.  \n_... Detecting ... {env.get('NUMBER_OF_PROCESSORS', 1)} worker available._  \n\n"
            'Run the app locally:  \n[Source code on Github](https://github.com/wujameszj/CourseProject)')


def main(debug):
    msg = st.empty()
    left, right = st.columns(2)
    left.header('Top2Vec'); right.header('LDA')
    
    
    with st.sidebar:
        st.subheader('Step 1: Choose dataset')    
        data = get_data(); sleep(3)  # give user time to correct input before start -- training cannot be stopped midway
    if not data: return   # invalid input; dont load rest of UI until new valid input is received 


    with left:
        t2v = train_top2vec(data)
        t2v_topics, _, __ = t2v.get_topics()


    with st.sidebar:
        st.subheader('Step 2: LDA parameters')
        lda_nTopic, passes, iters = get_param(len(t2v_topics))

    with right:
        if lda_nTopic:
            sleep(2); lda = MyLDA(data, lda_nTopic, passes, iters)
        
        
    with st.sidebar:
        st.subheader('Step 3: Compare topics and documents')
        _vocab = lda.vocab if lda_nTopic else t2v_topics
        keyword = selectbox('search by keyword', filter_keywords(t2v_topics, _vocab), help='This list consists of likely topic words in this dataset.')   # returns numpy_str

        st.write(AUTH_MSG)
               
    
    DEFAULT_EXAMPLE = 6
    with left:
        if keyword:
            topicIDs, docIDs = relevant_topics_docs(t2v, keyword, DEFAULT_EXAMPLE)
            msg.info(f'Displaying top {DEFAULT_EXAMPLE} documents related to "{keyword}".')
        else:
            nWordcloud = min(DEFAULT_EXAMPLE, len(t2v_topics))
            topicIDs, docIDs = range(nWordcloud), range(DEFAULT_EXAMPLE)
            msg.info(f'Displaying {DEFAULT_EXAMPLE} topics and documents.')
            
        create_wordcloud(t2v, topicIDs)
        display_doc(data, docIDs)

    
    with right:
        if lda_nTopic:
            if keyword:
                nDoc = min(DEFAULT_EXAMPLE, lda_nTopic)
                topicIDs, docIDs = lda.relevant_topics_docs(keyword, nDoc)              
            else:
                nWordcloud = min(DEFAULT_EXAMPLE, lda_nTopic)
                topicIDs, docIDs = range(nWordcloud), range(DEFAULT_EXAMPLE)
            
            create_wordcloud(lda.model, topicIDs)
            display_doc(data, docIDs)
            


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
        
    main(True)
    