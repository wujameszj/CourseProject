
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

#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


@st.cache
def extract_documents(url='https://cs.nyu.edu/~roweis/data/nips12raw_str602.tgz'):
    with smart_open.open(url, "rb") as file:
        with tarfile.open(fileobj=file) as tar:
            for member in tar.getmembers():
                if member.isfile() and re.search(r'nipstxt/nips\d+/\d+\.txt', member.name):
                    member_bytes = tar.extractfile(member).read()
                    yield member_bytes.decode('utf-8', errors='replace')
                    
@st.cache
def create_wordcloud(model, topic, nWords=30):
    # word_score_dict = dict(zip(
        # model.topic_words[topic],
        # softmax(self.topic_word_scores[topic_num])
    # ))
    
    wc = WordCloud(width=1600, height=400, background_color='black')
    if type(model) is LdaModel:
        wc = wc.generate_from_frequencies(dict(model.show_topic(topic, nWords)))
    else:
        word_prob = zip(
            model.topic_words[topic], 
            softmax(model.topic_word_scores[topic])
        )
        wc = wc.generate_from_frequencies(dict(word_prob))
    return wc.to_array()
    
    
@st.cache
def generate_LDA_model(data, n_topic):
    docs = list(extract_documents())[:99]

    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]
    docs = [[token for token in doc if len(token) > 2] for doc in docs]

    #nltk.download('wordnet')
    docs = [ [WordNetLemmatizer().lemmatize(token) for token in doc] for doc in docs]

    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=9, no_above=.5)   # Filter out words that occur in <20 docs or >50% of docs
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    _ = dictionary[0]  # This is only to "load" the dictionary.

    passes = 5  #20
    iterations = 100  #400
    model = LdaModel(
        corpus, n_topic, dictionary.id2token, chunksize=2000,
        alpha='auto', eta='auto', passes=passes, iterations=iterations, eval_every=None
    ) 
    return model


@st.cache
def generate_t2v_model(data, speed='learn'):
    if data == 'sklearn20news':
        return T2V.load('models/20news.model')
    else:
        return T2V(documents=newsgroups.data, speed=speed)

    
st.set_page_config('CS410 Project: Topic Modeling', layout="wide")

#st.title('CS410 Project: Topic Modeling')
left, right = st.columns(2)
left.header('top2vec'); right.header('LDA')

avail_data = ['arxiv', 'twitter', 'NYU/nips12raw_str602', 'reddit', 'sklearn20news']
dataset = st.sidebar.selectbox('dataset', avail_data, index=4, help='Choose dataset to perform topic modeling')

# avail_algo = ['top2vec', 'LDA']
# algo = st.sidebar.selectbox('algorithm', avail_algo, help='Choose algorithm to perform topic modeling')

if dataset=='arxiv':
    fromdate = st.sidebar.date_input('from date')
    start = st.sidebar.time_input('time')

n_topic_input = st.sidebar.empty()
topic_selection = st.sidebar.empty() 

run_t2v = st.sidebar.button('run top2vec')
#run_lda = st.sidebar.empty()

if run_t2v:
    t2v_model = generate_t2v_model(dataset)
    n_topic = t2v_model.get_num_topics()
    
    nOptions = 33 if 33 < n_topic else n_topic      # 33 is arbitrary
    topics, _, __ = t2v_model.get_topics(nOptions)
    topic_words = [words[0] for words in topics] 
    
    n_topic = n_topic_input.number_input('number of topics for LDA', 1, 9999, n_topic, help=f'{n_topic} is the number recommended by the top2vec algorithm for this dataset.')


    topic = topic_selection.selectbox('search topic by word', topic_words)
    
    img = [create_wordcloud(t2v_model, i) for i in range(3)]
    left.image(img)
    
    run_lda = st.sidebar.button('run LDA')
    
    if run_t2v and run_lda: #st.sidebar.button('run LDA'): 
        lda_model = generate_LDA_model(dataset, n_topic)
        img = [create_wordcloud(lda_model, i) for i in range(3)]
        right.image(img)


# _n = model.get_num_topics()
# ntopic = st.sidebar.number_input('number of topics', 1, 999, _n, 
#                                  help=f'{_n} is the recommended number determined by the algorithm, but you are free to modify it to see how it changes the topic model') 

# model.generate_topic_wordcloud(1)
# plt.savefig('data/temp.png')

#st.image('data/lda.svg', caption='wordcloud for topic 1')