
from bs4 import BeautifulSoup as BS
from requests import get 
import streamlit as st

from datetime import date, timedelta

from .misc import dwrite


base = 'https://en.wikipedia.org'
get_article = lambda url: BS(get(url).content, 'html.parser').get_text().strip()

@st.experimental_memo
def get_articles_from(mydate, _soup):
    _day = _soup.find(attrs={"aria-label": f"{mydate.strftime('%B')} {mydate.day}"})        # use double instead of single quote
    links = [a.get('href') for a in _day.find_all('a') if a.get('href').startswith('/wiki')]
    return [get_article(base+link) for link in set(links)]
    
    
#@st.experimental_memo    
def soup_of(mth, yr):
    link = f'/wiki/Portal:Current_events/{mth}_{yr}'
    return BS(get(base+link).content, 'html.parser')
    
    
@st.experimental_memo
def scrape(start, end, mydebug=True):
    
    articles, newMonth = [], True
    while(start <= end):
        if mydebug: dwrite(f'processing {start}\n')

        if newMonth: 
            soup = soup_of(start.strftime('%B'), start.year) 
        
        articles += get_articles_from(start, soup)
        
        # for link in links[:1]:
        #     soup = BS(get(base+link).content, 'html.parser')
        #     articles.append(soup.get_text(strip=True).replace('\n', ' '))
        
        start += timedelta(days=1)
        newMonth = True if start.day==1 else False
    
    if mydebug:
        dwrite(f'Collected {len(articles)} articles\n')
#        [dwrite(ar[:299] + '\n') for ar in articles[:2]]
        
    return articles



## speed up web scrape through parallelization