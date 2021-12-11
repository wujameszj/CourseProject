from bs4 import BeautifulSoup as BS
#import re 
from requests import get 
from datetime import date, timedelta

import streamlit as st


@st.experimental_memo
def scrape(start, end):
    base = 'https://en.wikipedia.org'    
    
    while(start<end):
        print(start)
        start.month
        
        soup = BS(get(base+url).content, 'html.parser')
        
        day = soup.find(attrs={"aria-label": "October 1"})
        links = [a.get('href') for a in day.find_all('a') if a.get('href').startswith('/wiki')]
        
        articles = []
        for link in links[:1]:
            soup = BS(get(base+link).content, 'html.parser')
            articles.append(soup.get_text(strip=True).replace('\n', ' '))
        
        start += timedelta(days=1)
    
    return articles
