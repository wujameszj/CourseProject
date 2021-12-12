
from streamlit import experimental_memo as st_cache

def dwrite(txt):
    with open('scrape.log', 'a', encoding='utf-8') as f:
        f.write(txt+'\n')

        
@st_cache
def filter_keywords(topics, minWords=30):
    wordsPerTopic = 1 if len(topics) > minWords else minWords//len(topics)
    return [None] + [words[i] for words in topics for i in range(wordsPerTopic) if len(words[i])>2] 
