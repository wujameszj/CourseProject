
from top2vec import Top2Vec as T2V
from gensim.models import LdaModel #, ldamulticore
from gensim.corpora import Dictionary

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer 

from time import time
import io, re, tarfile, smart_open, os.path

import streamlit as st
from wordcloud import WordCloud

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups as news

from numpy import array, argpartition as argp, argsort
from scipy.special import softmax

#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def extract_documents(url='https://cs.nyu.edu/~roweis/data/nips12raw_str602.tgz'):
    with smart_open.open(url, "rb") as file:
        with tarfile.open(fileobj=file) as tar:
            for member in tar.getmembers():
                if member.isfile() and re.search(r'nipstxt/nips\d+/\d+\.txt', member.name):
                    member_bytes = tar.extractfile(member).read()
                    yield member_bytes.decode('utf-8', errors='replace')
                    
@st.cache
def generate_LDA_model(data, nTopic, passes, iters):
    regex, lemma = RegexpTokenizer(r'\w+'), WordNetLemmatizer()
    docs = [regex.tokenize(doc.lower()) for doc in data['data']]
    docs = [[token for token in doc if len(token) > 1 and not token.isnumeric()] for doc in docs]
    #docs = [[lemma.lemmatize(token) for token in doc] for doc in docs]

    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=2, no_above=.8, keep_n=None)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    _ = dictionary[0]  # This is only to "load" the dictionary.
    
    model = LdaModel(
        corpus, nTopic, dictionary.id2token, chunksize=99999,
        alpha='auto', eta='auto', passes=passes, iterations=iters, eval_every=None, update_every=0
    ) 
    doc_topic_prob = [model[doc] for doc in corpus]         # equivalent to get_document_topics()
    return model, list(dictionary.values()), doc_topic_prob

@st.experimental_memo
def calc_relevance(doc_topic_prob, topicID):
    return array( 
        [sum([p if i==topicID else 0 for i,p in doc]) for doc in doc_topic_prob] 
    )
    
#calc_relevance = lambda doc_topic_prob, topicID: [sum([p if i==topicID else 0 for i,p in doc]) for doc in doc_topic_prob]


@st.cache(allow_output_mutation=True)
def generate_t2v_model(data, speed='learn'):
    if data['name'] == 'sklearn20news':
        return T2V.load('models/20news.model')
    else:
        return T2V(documents=data['data'], speed=speed, min_count=50, keep_documents=False)

@st.cache(allow_output_mutation=True)
def retrieve(dataset, fromDate=None, toDate=None):    
    if dataset == 'sklearn20news':
        return news(subset='all', remove=('headers', 'footers', 'quotes')).data
   
@st.experimental_memo
def _create_wordcloud(word_prob):
    wc = WordCloud(width=1600, height=400, background_color='black')
    return wc.generate_from_frequencies(dict(word_prob)).to_array()

def create_wordcloud(model, topicIDs, nWords=30):
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
    
    
def main():
    st.set_page_config('CS410 Project', layout="wide")
    st.title('Compare Topic Modeling Algorithms')
    patient = st.empty()
    msg = st.empty()
    left, right = st.columns(2)
    left.header('top2vec'); right.header('LDA')
    

    avail_data = ['arxiv', 'twitter', 'NYU/nips12raw_str602', 'reddit', 'sklearn20news']
    with st.sidebar:
        dataset = st.selectbox('dataset', avail_data, index=4, help='Choose dataset to perform topic modeling')
        if dataset == 'arxiv':
            fromdate = st.date_input('from date')
            start = st.time_input('time')
        elif dataset == 'sklearn20news':
            data = {'name': 'sklearn20news', 'data': retrieve(dataset)}


    t2v_model = generate_t2v_model(data)

    nTopic = t2v_model.get_num_topics()
    DEFAULT_WORDCLOUD = 5
    nExample = DEFAULT_WORDCLOUD if DEFAULT_WORDCLOUD < nTopic else nTopic 
    
    topics, _, __ = t2v_model.get_topics(nTopic)
    topic_words = [None] + [words[0] for words in topics] 

    with st.sidebar:
        st.subheader('LDA parameters')
        lda_param = st.expander('optional training parameters for LDA model')
        with lda_param:
            nTopic = int(st.number_input('number of topics', 0, 999, 0, help=f'A larger number increases computation time. Based on Top2vec, we recommend {int(nTopic/1.5)} for this dataset.'))
            passes = int(st.number_input('passes', 1, 99, 2, help='Higher number increases model quality at the cost of computation time.'))
            iters = int(st.number_input('iterations', 1, 999, 50, help='Higher number increases model quality at the cost of computation time.'))
        topic = st.selectbox('search topic by word', topic_words, help='This list consists of likely topic words in this dataset.')

    
    with left:
        if topic is None:
            msg.info(f'Displaying {nExample} unrelated topics and documents.')
            topicIDs, docIDs = range(nExample), range(nExample)
        else:
            msg.info(f'Displaying {nExample} topics and documents related to "{topic}".')
            _,_,_, topicIDs = t2v_model.query_topics(str(topic), nExample)
            _,_, docIDs = t2v_model.search_documents_by_keywords([topic], nExample, keywords_neg=None, return_documents=True, use_index=False, ef=None)
        st.image(create_wordcloud(t2v_model, topicIDs))
        st.table([data['data'][i] for i in docIDs])
    
    
    if nTopic:
        with right:
           # patient.info(f'Training model with {nTopic} topics for {passes} passes and {iters} iterations ... could take more than {nTopic*passes*iters//99} minutes. Please be patient.')
            patient.info(f'Training model with {nTopic} topics for {passes} passes and {iters} iterations. Please be patient.')
            lda_model, dictionary, doc_topic_prob = generate_LDA_model(data, nTopic, passes, iters)
            
            if topic is None:
                topicIDs, docIDs = range(nExample), range(nExample, nExample*2)
            else:
                topicIDs = lda_model.get_term_topics(dictionary.index(topic), minimum_probability=0)
                topicIDs = [i for i,_ in topicIDs[:nExample]]
                
                doc_prob = calc_relevance(doc_topic_prob, topicIDs[0])
                docIDs = argp(doc_prob, -nExample)[-nExample:]
                docIDs = docIDs[ argsort(doc_prob[docIDs])[::-1] ]    # list largest first
                
            st.image(create_wordcloud(lda_model, topicIDs))
            st.table([data['data'][i] for i in docIDs])


    patient.empty()
    st.sidebar.write('[Source code on Github](https://github.com/wujameszj/CourseProject)')


if __name__ == '__main__':
    #nltk.download('wordnet')
    
    main()