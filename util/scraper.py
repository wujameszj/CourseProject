
from bs4 import BeautifulSoup as BS
from requests import get 

import streamlit as st

from time import time
from datetime import date, timedelta

from .misc import dwrite



base = 'https://en.wikipedia.org'


remove_foreign = lambda doc: ' '.join([word for word in doc.split(' ') if word.isascii()])

def clean(doc, multi_lang):
    doc = doc[15 + doc.index('Jump to search\n'):]
    return doc if multi_lang else remove_foreign(doc)


@st.experimental_memo 
def get_article(url, multi_lang=False):
    try:
        article = BS(get(base+url).content, 'html.parser').get_text().strip()
        return clean(article, multi_lang)
    except Exception as e:
        dwrite(f'Exception {url}: {repr(e)}')          
        return ''


    
@st.experimental_memo 
def get_links_on(mydate, _soup):
    
    _day = _soup.find(attrs={'aria-label': f"{mydate.strftime('%B')} {mydate.day}"})  
    links = [a.get('href') for a in _day.find_all('a') if a.get('href').startswith('/wiki')]
    
    # for link in links[:1]:
    #     soup = BS(get(base+link).content, 'html.parser')
    #     articles.append(soup.get_text())
    return links
    
    
    
#@st.experimental_memo    
def soup_of(mth, yr):
    
    link = f'/wiki/Portal:Current_events/{mth}_{yr}'
    try:
        soup = BS(get(base+link).content, 'html.parser')
        return soup
    
    except Exception as e:
        dwrite(f'Exception {yr} {mth}: {repr(e)}')
        st.experimental_rerun()
            
    
    
@st.experimental_memo 
def scrape(start, end):
    t = time()    
    
    links, newMonth = [], True
    while(start <= end):
        if newMonth: 
            soup = soup_of(start.strftime('%B'), start.year)         
        links += get_links_on(start, soup)

        start += timedelta(days=1)
        newMonth = True if start.day==1 else False
        
    articles = [get_article(link) for link in set(links)]

    dwrite(f'{start} Collected {len(articles)} articles in {int(time()-t)} sec\n')
    
    return articles
