
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
from scipy.special import softmax
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups as news

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
def generate_LDA_model(data, n_topic, passes, iters):
    docs = data['data'] #list(extract_documents())[:99]

    # if type(docs[0]) is str:         # streamlit caches previous runs, wherein docs is already a list of token lists  
    reg = RegexpTokenizer(r'\w+')
    docs = [reg.tokenize(doc.lower()) for doc in docs]
    docs = [[token for token in doc if (len(token) > 1) and (not token.isnumeric())] for doc in docs] 
    #nltk.download('wordnet')
    lemma = WordNetLemmatizer()
    docs = [[lemma.lemmatize(token) for token in doc] for doc in docs]

    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=5, no_above=.5, keep_n=100000)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    _ = dictionary[0]  # This is only to "load" the dictionary.
    
    model = LdaModel(
        corpus, n_topic, dictionary.id2token, chunksize=99999,
        alpha='auto', eta='auto', passes=passes, iterations=iters, eval_every=None
    ) 
    return model, list(dictionary.values())


@st.cache(allow_output_mutation=True)
def generate_t2v_model(data, speed='learn'):
    if data['name'] == 'sklearn20news':
        return T2V.load('models/20news.model')
    else:
        return T2V(documents=data['data'], speed=speed)

@st.cache(allow_output_mutation=True)
def retrieve(dataset, fromDate=None, toDate=None):    
    if dataset == 'sklearn20news':
        return news(subset='all', remove=('headers', 'footers', 'quotes')).data
                        
#@st.cache
def create_wordcloud(column, model, topicIDs, nWords=30):    
    #canvas = column.empty()
    
    wc_list = []
    for topicID in topicIDs:
        wc = WordCloud(width=1600, height=400, background_color='black')
        if type(model) is LdaModel:
            word_prob = model.show_topic(topicID, nWords)
        elif type(model) is T2V:
            word_prob = zip(
                model.topic_words[topicID], 
                softmax(model.topic_word_scores[topicID])
            )
        wc = wc.generate_from_frequencies(dict(word_prob))
        wc_list.append(wc.to_array())
        # canvas.image(wc_list)
    column.image(wc_list)
            
    
def main():
    st.set_page_config('CS410 Project', layout="wide")
    st.title('Compare Topic Modeling Algorithms')
    patient_msg = st.empty()
    left, right = st.columns(2)
    left.header('top2vec'); right.header('LDA')

    avail_data = ['arxiv', 'twitter', 'NYU/nips12raw_str602', 'reddit', 'sklearn20news']
    dataset = st.sidebar.selectbox('dataset', avail_data, index=4, help='Choose dataset to perform topic modeling')
    if dataset == 'arxiv':
        fromdate = st.sidebar.date_input('from date')
        start = st.sidebar.time_input('time')
    elif dataset == 'sklearn20news':
        data = {'name': 'sklearn20news', 'data': retrieve(dataset)}


    patient_msg.info('LDA could take more than 10 minutes. Please be patient.')
    t2v_model = generate_t2v_model(data)


    n_topic = t2v_model.get_num_topics()
    topics, _, __ = t2v_model.get_topics(n_topic)
    topic_words = [None] + [words[0] for words in topics] 

    with st.sidebar:
        lda_param = st.expander('training parameters for LDA model')
        with lda_param:
            n_topic = int(st.number_input('number of topics', 1, 999, n_topic//2, help='A larger number increases computation time.'))
            passes = int(st.number_input('passes', 1, 99, 3, help='A larger number increases computation time.'))
            iters = int(st.number_input('iterations', 1, 999, 50, help='A larger number increases computation time.'))
        topic = st.selectbox('search topic by word', topic_words, help='This list consists of likely topic words in this dataset.')


    DEFAULT_WORDCLOUD = 5 
    nWordcloud = DEFAULT_WORDCLOUD if DEFAULT_WORDCLOUD < n_topic else n_topic 

    if topic is None:
        topicID = range(nWordcloud)
    else:
        _,_,_, topicID = t2v_model.query_topics(str(topic), nWordcloud)
    create_wordcloud(left, t2v_model, topicID)

    lda_model, dictionary = generate_LDA_model(data, n_topic, passes, iters)

    if topic is None:
        topicID = range(nWordcloud)
    else:
        topicID = lda_model.get_term_topics(dictionary.index(topic), 0)
        topicID = [i for i,_ in topicID[:nWordcloud]]
    create_wordcloud(right, lda_model, topicID)


    patient_msg.empty()


if __name__ == '__main__':
    main()