import pandas as pd
import yfinance as yf
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import streamlit as st

# API setup
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    df = yf.download(symbol, start=start_date, end=end_date)
    return df

@st.cache_data(ttl=3600)
def get_news_headlines(symbol):
    try:
        news = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5)
        return [(article['title'], article['description'], article['url']) for article in news['articles']]
    except Exception as e:
        print(f"News API error: {str(e)}")
        return []