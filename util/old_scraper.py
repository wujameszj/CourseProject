
# This old scraper deduplicates links daily instead of for all links which the new scraper does.
# However, this scrape caches daily runs which may be more efficient in some cases where some dates in a given range have already been processed.



from bs4 import BeautifulSoup as BS
from requests import get 

import streamlit as st

from time import time
from datetime import date, timedelta

from .misc import dwrite



base = 'https://en.wikipedia.org'


remove_foreign = lambda doc: ' '.join([word for word in doc.split(' ') if word.isascii()])

def clean(doc, multi_lang=False):
    doc = doc[15 + doc.index('Jump to search\n'):]
    return doc if multi_lang else remove_foreign(doc)


get_article = lambda url: BS(get(base+url).content, 'html.parser').get_text().strip()

@st.experimental_memo 
def get_articles_from(mydate, _soup):
    
    _day = _soup.find(attrs={'aria-label': f"{mydate.strftime('%B')} {mydate.day}"})
    
    links = [a.get('href') for a in _day.find_all('a') if a.get('href').startswith('/wiki')]
    
    # for link in links[:1]:
    #     soup = BS(get(base+link).content, 'html.parser')
    #     articles.append(soup.get_text())
            
    try:
        return [clean(get_article(link)) for link in set(links)]        
    except Exception as e:
        dwrite(f'Exception {mydate}: {repr(e)}')    
        st.experimental_rerun()

    
    
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
    
    articles, newMonth = [], True
    while(start <= end):
        if newMonth: 
            soup = soup_of(start.strftime('%B'), start.year)         

        t = time()    
        articles += get_articles_from(start, soup)

        dwrite(f'{start} Collected {len(articles)} articles in {time()-t} sec\n')
        print(f'{start} Collected {len(articles)} articles in {time()-t} sec\n')
#        [dwrite(ar[:299] + '\n') for ar in articles[:2]]

        start += timedelta(days=1)
        newMonth = True if start.day==1 else False
        
    return articles
