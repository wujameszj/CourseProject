from wordcloud import WordCloud
import streamlit as st

from top2vec import Top2Vec
from gensim.models import LdaModel
from numpy.random import random


@st.experimental_memo
def _create_wordcloud(word_prob):
    wc = WordCloud(width=1600, height=400, background_color='black')
    return wc.generate_from_frequencies(dict(word_prob)).to_array()


def create_wordcloud(model, topicIDs, nWords=22):
    with st.container():
        for topicID in topicIDs:
            if type(model) is LdaModel:
                word_prob = model.show_topic(topicID, nWords)
            elif type(model) is Top2Vec:
                word_prob = zip(
                    model.topic_words[topicID][:nWords], 
                    model.topic_word_scores[topicID][:nWords]
                )
            st.image(_create_wordcloud(word_prob))
    
    
def display_doc(data, docIDs):
    docs = [data['data'][i] for i in docIDs] 
    for doc in docs:
        doc = doc.strip()
        n = doc.count('\n') * 30  # pixels per line
        st.text_area('', doc, height=400 if n > 400 else n, key=random(), help='You can adjust the height of the text display by dragging from the bottom-right corner.')
    