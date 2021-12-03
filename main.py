
from gensim.models import LdaModel #, ldamulticore
from gensim.corpora import Dictionary
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer 
from top2vec import Top2Vec as T2V

import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from numpy import array, argmax, argpartition as argp, argsort
from numpy.random import random
from pandas import DataFrame as DF
from scipy.special import softmax
from sklearn.datasets import fetch_20newsgroups as news

from os import environ


@st.experimental_memo #@st.cache #(allow_output_mutation=True)
def train_LDA_model(data, nTopic, passes, iters):
    regex, lemma = RegexpTokenizer(r'\w+'), WordNetLemmatizer()
    docs = [regex.tokenize(doc.lower()) for doc in data['data']]
    docs = [[token for token in doc if len(token) > 1 and not token.isnumeric()] for doc in docs]
    #docs = [[lemma.lemmatize(token) for token in doc] for doc in docs]

    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=2, no_above=.5, keep_n=None)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    _ = dictionary[0]  # This is only to "load" the dictionary.
    
    model = LdaModel(
        corpus, nTopic, dictionary.id2token, chunksize=environ.get('CHUNK', 99999), passes=passes, iterations=iters, update_every=1, 
        alpha='auto', eta='auto', minimum_probability=0, eval_every=None
    ) 
#    doc_topic_prob = [model[doc] for doc in corpus]         # equivalent to get_document_topics()
    return model, list(dictionary.values()), corpus #doc_topic_prob

@st.experimental_memo
def calc_relevance(corpus, wordID):
    return array( 
        [sum([n if i==wordID else 0 for i,n in doc]) for doc in corpus]
    )

# def calc_relevance(doc_topic_prob, topicID):
#     return array( 
#         [sum([p if i==topicID else 0 for i,p in doc]) for doc in doc_topic_prob] 
#     )

# calc_relevance = lambda doc_topic_prob, topicID: [sum([p if i==topicID else 0 for i,p in doc]) for doc in doc_topic_prob]


@st.cache(allow_output_mutation=True)
def train_t2v_model(data, speed='learn'):
    if data['name'] == 'sklearn20news':
        return T2V.load('models/20news.model')
    else:
        return T2V(documents=data['data'], speed=speed, min_count=9, keep_documents=False, workers=environ.get('NUMBER_OF_PROCESSORS', 1))

@st.cache(allow_output_mutation=True)
def retrieve(dataset, fromDate=None, toDate=None):    
    if dataset == 'sklearn20news':
        return news(subset='all', remove=('headers', 'footers', 'quotes')).data
    
   
@st.experimental_memo
def _create_wordcloud(word_prob):
    wc = WordCloud(width=1600, height=400, background_color='black')
    return wc.generate_from_frequencies(dict(word_prob)).to_array()

def __create_wordcloud(model, topicIDs, nWords=30):
    #canvas = column.empty()
    wc_list = []
    for topicID in topicIDs:
        if type(model) is LdaModel:
            word_prob = model.show_topic(topicID, nWords)
        elif type(model) is T2V:
            word_prob = zip(
                model.topic_words[topicID], 
                softmax(model.topic_word_scores[topicID])
            )
        wc_list.append(_create_wordcloud(word_prob))
        # canvas.image(wc_list)
    return wc_list

def create_wordcloud(model, topicIDs, nWords=30):
    with st.container():
        for topicID in topicIDs:
            if type(model) is LdaModel:
                word_prob = model.show_topic(topicID, nWords)
            elif type(model) is T2V:
                word_prob = zip(
                    model.topic_words[topicID], 
                    softmax(model.topic_word_scores[topicID])
                )
            st.image(_create_wordcloud(word_prob))


@st.experimental_memo(suppress_st_warning=True)
def display_doc(docs):
    for i, doc in enumerate(docs):
        # with st.expander(f'Doc {i+1}', True):
        doc = doc.strip()
        n = doc.count('\n') * 38  # approx 35 pixels per line
        st.text_area('', doc, height=500 if n > 500 else n, key=random())
#            st.dataframe(DF([doc]), height=500)
#            st.text(doc)
    

PASS_MSG = 'Number of passes through corpus, i.e., passes per mini-batch. \nHigher number may improve model by facilitating convergence for small corpora at the cost of computation time.'
ITER_MSG = 'Number of E-step per document per pass. Higher number may improve model by fascilitating document convergence at the cost of computation time.'


def main():
    st.set_page_config('CS410 Project', layout="wide")
    st.title('Compare Topic Modeling Algorithms')
    # patient = st.empty()
    msg = st.empty()
    left, right = st.columns(2)
    left.header('top2vec'); right.header('LDA')
    
    WORKER, CHUNKSIZE = environ.get('NUMBER_OF_PROCESSORS', 1), environ.get('CHUNK', 99999)
    

    avail_data = ['arxiv', 'twitter', 'NYU/nips12raw_str602', 'reddit', 'sklearn20news']
    with st.sidebar:
        st.subheader('Step 1:')
        dataset = st.selectbox('dataset', avail_data, index=4, help='Choose dataset to perform topic modeling')
        if dataset == 'arxiv':
            fromdate = st.date_input('from date')
            start = st.time_input('time')
        elif dataset == 'sklearn20news':
            data = {'name': 'sklearn20news', 'data': retrieve(dataset)}


    t2v_model = train_t2v_model(data)

    nTopic = t2v_model.get_num_topics()
    DEFAULT_EXAMPLE = 3
    nExample = DEFAULT_EXAMPLE if DEFAULT_EXAMPLE < nTopic else nTopic 
    
    topics, _, __ = t2v_model.get_topics(nTopic)
    topic_words = [None] + [words[0] for words in topics] 

    with st.sidebar:
        st.subheader('Step 2: LDA parameters')
        nTopic = int(st.number_input(
            'number of topics', 0, 999, 0, help=f'Larger number increases computation time. Based on Top2vec, we recommend {int(nTopic*.7)} for this dataset.'))
        optional = st.expander('optional training parameters')
        with optional:    
            passes = int(st.number_input('passes', 1, 99, 1, help=PASS_MSG))
            iters = int(st.number_input('iterations', 1, 999, 50, help=ITER_MSG))
        st.subheader('Step 3: Compare topics and documents')
        topic = st.selectbox('search by keyword', topic_words, help='This list consists of likely topic words in this dataset.')   # returns numpy_str
        
        st.write('_ _ _\n**Shoutout to Streamlit for generously hosting this app for free! \U0001f600**  \n- - -\n')
        st.write(f'App is sluggish? Sorry about that.  \n_Detecting... {WORKER} worker available._')
        st.write('Run the app locally:  \n[Source code on Github](https://github.com/wujameszj/CourseProject)')

    
    with left:
        if topic:
            msg.info(f'Displaying top 6 documents related to "{topic}".')
            _,_,_, topicIDs = t2v_model.query_topics(str(topic), 1)         # topic is actually of type numpy_str which top2vec doesnt accept but LDA (gensim) does
            _, docIDs = t2v_model.search_documents_by_keywords([topic], nExample*2, keywords_neg=None, return_documents=False, use_index=False, ef=len(data['data']))
        else:
            msg.info(f'Displaying {nExample*2} unrelated topics and documents.')
            topicIDs, docIDs = range(nExample*2), range(nExample*2)
        create_wordcloud(t2v_model, topicIDs)
#        st.image(create_wordcloud(t2v_model, topicIDs))
        display_doc( [data['data'][i] for i in docIDs] )
                
    
    if nTopic:
        with right:
            patient = st.info(f'Training model with {nTopic} topics for {passes} passes and {iters} iterations. Please be patient.')
#            lda_model, dictionary, doc_topic_prob = generate_LDA_model(data, nTopic, passes, iters)
            lda_model, dictionary, corpus = train_LDA_model(data, nTopic, passes, iters)
           
            if topic:
                topic_prob = lda_model.get_term_topics(dictionary.index(topic), minimum_probability=0)
                idx = argmax([p for i,p in topic_prob])
                topicIDs = [ topic_prob[idx][0] ]
                
#                topicIDs = [i for i,p in topicIDs[:1]]
 #               doc_prob = calc_relevance(doc_topic_prob, topicIDs[0])
                doc_prob = calc_relevance(corpus, dictionary.index(topic))
                docIDs = argp(doc_prob, -nExample*2)[-nExample*2:]
                docIDs = docIDs[ argsort(doc_prob[docIDs])[::-1] ]    # list largest first                
            else:
                topicIDs, docIDs = range(nExample*2), range(nExample*2, nExample*4)
            patient.empty()
            
            create_wordcloud(lda_model, topicIDs)
#            st.image(create_wordcloud(lda_model, topicIDs))
            display_doc( [data['data'][i] for i in docIDs] )        



if __name__ == '__main__':
    #nltk.download('wordnet')
    
    main()
    