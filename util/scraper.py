from bs4 import BeautifulSoup as BS
#import re 
from requests import get 
from datetime import date, timedelta

import streamlit as st




#def collect_links(start, end)

base = 'https://en.wikipedia.org'
get_article = lambda url: BS(get(url).content, 'html.parser').get_text().strip().replace('\n', ' ')

#@st.experimental_memo
def scrape(start, end, mydebug=True, dailyLimit=9999):

    
    articles, newMonth = [], True
    while(start < end):
        if mydebug:
            with open('scrape.log', 'a') as f:
                f.write(f'processed {start}\n')
                
        if newMonth:
            link = '/wiki/Portal:Current_events/{}_{}'.format(start.strftime('%B'), start.year)
            monthSoup = BS(get(base+link).content, 'html.parser')
        
        _day = monthSoup.find(attrs={"aria-label": f"{start.strftime('%B')} {start.day}"})      # use double instead of single quote
        links = [a.get('href') for a in _day.find_all('a') if a.get('href').startswith('/wiki')]
        
        
        articles += [get_article(base+link) for link in links]
#        articles += [BS(get(base+link).content, 'html.parser').get_text().strip().replace('\n', ' ') for link in links]
        
        # for link in links[:1]:
        #     soup = BS(get(base+link).content, 'html.parser')
        #     articles.append(soup.get_text(strip=True).replace('\n', ' '))
        
        start += timedelta(days=1)
        newMonth = True if start.day==1 else False
    
    st.write(len(articles))
    if mydebug:
        with open('scrape.log', 'a') as f:
            f.write(f'Collected {len(articles)} articles\n')
            f.write(articles[0][:99] + '\n')
    return articles
