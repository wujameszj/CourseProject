
from bs4 import BeautifulSoup as BS
from requests import get 

import streamlit as st
from datetime import date, timedelta

from .misc import dwrite



base = 'https://en.wikipedia.org'
get_article = lambda url: BS(get(base+url).content, 'html.parser').get_text().strip()
clean = lambda doc: doc[15 + doc.index('Jump to search\n'):]



@st.experimental_memo 
def get_articles_from(mydate, _soup):
    
    _day = _soup.find(attrs={"aria-label": f"{mydate.strftime('%B')} {mydate.day}"})        # use double instead of single quote
    links = [a.get('href') for a in _day.find_all('a') if a.get('href').startswith('/wiki')]
    
    try:
        art = [clean(get_article(link)) for link in set(links)]
    except Exception as e:
        dwrite(f'Exception {mydate}: {repr(e)}')    
    return art

    
    
#@st.experimental_memo    
def soup_of(mth, yr):
    
    link = f'/wiki/Portal:Current_events/{mth}_{yr}'
    try:
        soup = BS(get(base+link).content, 'html.parser')
    except Exception as e:
        dwrite(f'Exception {yr} {mth}: {repr(e)}')
        
    return soup
    
    
    
@st.experimental_memo 
def scrape(start, end):
    
    articles, newMonth = [], True
    while(start <= end):
        if newMonth: 
            soup = soup_of(start.strftime('%B'), start.year) 
        
        articles += get_articles_from(start, soup)
        
        # for link in links[:1]:
        #     soup = BS(get(base+link).content, 'html.parser')
        #     articles.append(soup.get_text(strip=True).replace('\n', ' '))
        
        start += timedelta(days=1)
        newMonth = True if start.day==1 else False
    
        dwrite(f'{start} Collected {len(articles)} articles\n');  print(f'{start} Collected {len(articles)} articles\n')
#        [dwrite(ar[:299] + '\n') for ar in articles[:2]]
        
    return articles
