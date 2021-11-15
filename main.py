import streamlit as st
from top2vec import Top2Vec as T2V

import matplotlib.pyplot as plt


st.title('CS410 Project: Topic Modeling')

avail_data = ['arxiv', 'twitter', 'cbc', 'reddit', 'sklearn20news']
dataset = st.sidebar.selectbox('dataset', avail_data, index=4, help='Choose dataset to perform topic modeling')

avail_algo = ['top2vec', 'LDA']
algo = st.sidebar.selectbox('algorithm', avail_algo, help='Choose algorithm to perform topic modeling')


if dataset=='arxiv':
    date = st.sidebar.date_input('date')
    start = st.sidebar.time_input('time')

@st.cache
def generate_model():
    model = T2V.load('models/20news.model')
    return model

model = generate_model()
_n = model.get_num_topics()
ntopic = st.sidebar.number_input('number of topics', 1, 999, _n, 
                                 help=f'{_n} is the recommended number determined by the algorithm, but you are free to modify it to see how it changes the topic model') 

model.generate_topic_wordcloud(1)
plt.savefig('data/temp.png')
st.image('data/temp.png', caption='wordcloud for topic 1')