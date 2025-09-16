import time
import logging
from googlesearch import search
from urllib.parse import urlparse
import re

import time
import logging
from googlesearch import search
from urllib.parse import urlparse
import re

import time
import logging
from googlesearch import search
from urllib.parse import urlparse
import re

def _fetch_headlines(queries, headline_limit=20, pause_between_queries=2, num_results_per_query=7):
    """
    Generic function to fetch news headlines for a list of queries.
    
    Args:
        queries (list): List of search queries
        headline_limit (int): Maximum number of headlines to return
        pause_between_queries (int): Seconds to pause between Google queries
        num_results_per_query (int): Number of results to fetch per query
    
    Returns:
        list: List of unique headlines
    """
    headlines = []
    
    try:
        for query in queries:
            if len(headlines) >= headline_limit:
                break
                
            try:
                # Perform search with correct parameters for current googlesearch version
                search_results = search(
                    query, 
                    num_results=num_results_per_query,  # Correct parameter name
                    advanced=False,  # Get simple URLs instead of advanced results
                    sleep_interval=pause_between_queries  # Correct parameter name
                )
                
                for url in search_results:
                    if len(headlines) >= headline_limit:
                        break
                    
                    # Extract headline from URL more robustly
                    headline = _extract_headline_from_url(url)
                    if headline and headline not in headlines:
                        headlines.append(headline)
                        
            except Exception as query_error:
                logging.warning(f"Query '{query}' failed: {query_error}")
                continue
                
        return headlines[:headline_limit]  # Ensure we don't exceed limit
        
    except Exception as e:
        logging.error(f"Critical error in headline fetching: {e}")
        return []

def _extract_headline_from_url(url):
    """Extract a meaningful headline from a URL."""
    try:
        # Parse URL properly
        parsed_url = urlparse(url)
        path = parsed_url.path
        
        # Extract the last part of the path
        last_segment = path.split('/')[-1]
        
        # Clean up the headline
        headline = last_segment.replace('-', ' ').replace('_', ' ').title()
        
        # Remove common file extensions and parameters
        headline = re.sub(r'\.(html|php|aspx?|jsp)$', '', headline, flags=re.IGNORECASE)
        headline = re.sub(r'\?.*$', '', headline)
        
        # Remove any remaining special characters but keep spaces
        headline = re.sub(r'[^\w\s]', '', headline)
        headline = re.sub(r'\s+', ' ', headline).strip()
        
        return headline if len(headline) > 5 else None  # Filter out very short headlines
        
    except Exception as e:
        logging.debug(f"Failed to extract headline from {url}: {e}")
        return None
    
def _analyze_macro_news():
    """Analyzes sentiment of macro-economic news."""
    queries = [
        '"India GDP" OR "inflation CPI" OR "IIP data" latest',
        '"RBI monetary policy" OR "repo rate" OR "interest rates" decision',
        '"India trade deficit" OR "exports imports" OR "FDI FII flows" data',
    ]
    headlines = _fetch_headlines(queries, 15)
    analysis['macro_news_headlines'] = headlines
    if not headlines:
        analysis['macro_news_sentiment'] = "No recent macro news found."
        score_breakdown['macro_news_score'] = 0
        return

    avg_sentiment = sum(vader.polarity_scores(h)['compound'] for h in headlines) / len(headlines)
# Based on VADER paper recommendations
    if avg_sentiment >= 0.5:          # Strong positive
        sentiment_text, news_score = "Very Positive", 5
    elif avg_sentiment >= 0.05:        # Moderate positive  
        sentiment_text, news_score = "Positive", 2
    elif avg_sentiment <= -0.5:       # Strong negative
        sentiment_text, news_score = "Very Negative", -5
    elif avg_sentiment <= -0.05:       # Moderate negative
        sentiment_text, news_score = "Negative", -2
    else:                             # Neutral zone
        sentiment_text, news_score = "Neutral", 0
    analysis['macro_news_sentiment'] = f"Overall sentiment is {sentiment_text} (Score: {avg_sentiment:.2f})"
    score_breakdown['macro_news_score'] = news_score
    score += news_score

    prompt = "Summarize the following macro-economic news headlines for India in 2-3 key bullet points.\n\nHeadlines:\n" + "\n".join(headlines)
    
_analyze_macro_news()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
import streamlit as st
import yfinance as yf
import pandas as pd
import logging
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from groq import Groq
from googlesearch import search
ticker_symbol = "SBIN.NS"
import time
ticker = yf.Ticker(ticker_symbol)
data = None
analysis = {}
score = 0
score_breakdown = {}
info = ticker.info
vader = SentimentIntensityAnalyzer()
try:
    if os.getenv("GROQ_API_KEY"):
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    else:
        groq_client = None
except Exception as e:
    logging.error(f"Failed to initialize Groq client: {e}")
    groq_client = None
    

def _groq_summary(self, prompt, summary_key):
    """Generates a summary using the Groq API."""
    if self.groq_client:
        try:
            logging.info(f"Generating AI summary for {summary_key}...")
            chat_completion = self.groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.1-8b-instant")
            self.analysis[summary_key] = chat_completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Failed to generate Groq summary for {summary_key}: {e}")
            self.analysis[summary_key] = "AI summary could not be generated."
    else:
        self.analysis[summary_key] = "Groq API key not configured. AI summary unavailable."

def _analyze_company_news(self):
    """Analyzes sentiment of company-specific news."""
    company_name = self.info.get('longName', self.ticker_symbol)
    queries = [
        f'"{company_name}" latest quarterly results OR revenue OR profit OR guidance',
        f'"{company_name}" merger OR acquisition OR partnership OR deal OR leadership change OR new product',
        f'"{company_name}" regulatory news OR legal issues OR analyst rating',
    ]
    headlines = self._fetch_headlines(queries, 15)
    self.analysis['company_news_headlines'] = headlines 
    if not headlines:
        self.analysis['company_news_sentiment'] = "No recent news found."
        self.score_breakdown['company_news_score'] = 0
        return
    
    avg_sentiment = sum(self.vader.polarity_scores(h)['compound'] for h in headlines) / len(headlines)
    # Based on VADER paper recommendations
    if avg_sentiment >= 0.5:          # Strong positive
        sentiment_text, news_score = "Very Positive", 10
    elif avg_sentiment >= 0.05:        # Moderate positive  
        sentiment_text, news_score = "Positive", 5
    elif avg_sentiment <= -0.5:       # Strong negative
        sentiment_text, news_score = "Very Negative", -10
    elif avg_sentiment <= -0.05:       # Moderate negative
        sentiment_text, news_score = "Negative", -5
    else:                             # Neutral zone
        sentiment_text, news_score = "Neutral", 0
    self.analysis['company_news_sentiment'] = f"Overall sentiment is {sentiment_text} (Score: {avg_sentiment:.2f})"
    self.score_breakdown['company_news_score'] = news_score
    self.score += news_score
    
    prompt = f"Summarize the following news headlines for {company_name} in 2-3 key bullet points.\n\nHeadlines:\n" + "\n".join(headlines)
    self._groq_summary(prompt, 'company_news_summary')

def _analyze_sector_news(self):
    """Analyzes sentiment of sector-specific news."""
    sector = self.info.get('sector') # Default to banking if sector not found
    queries = [
        f'"{sector} sector" India outlook OR growth OR challenges OR trends',
        f'"{sector} industry" India regulatory changes OR RBI policy OR guidelines',
    ]
    headlines = self._fetch_headlines(queries, 15)
    self.analysis['sector_news_headlines'] = headlines
    if not headlines:
        self.analysis['sector_news_sentiment'] = "No recent sector news found."
        self.score_breakdown['sector_news_score'] = 0
        return

    avg_sentiment = sum(self.vader.polarity_scores(h)['compound'] for h in headlines) / len(headlines)
# Based on VADER paper recommendations
    if avg_sentiment >= 0.5:          # Strong positive
        sentiment_text, news_score = "Very Positive", 5
    elif avg_sentiment >= 0.05:        # Moderate positive  
        sentiment_text, news_score = "Positive", 3
    elif avg_sentiment <= -0.5:       # Strong negative
        sentiment_text, news_score = "Very Negative", -5
    elif avg_sentiment <= -0.05:       # Moderate negative
        sentiment_text, news_score = "Negative", -3
    else:                             # Neutral zone
        sentiment_text, news_score = "Neutral", 0
    self.analysis['sector_news_sentiment'] = f"Overall sentiment is {sentiment_text} (Score: {avg_sentiment:.2f})"
    self.score_breakdown['sector_news_score'] = news_score
    self.score += news_score

    prompt = f"Summarize the following news headlines for the Indian {sector} sector in 2-3 key bullet points.\n\nHeadlines:\n" + "\n".join(headlines)
    self._groq_summary(prompt, 'sector_news_summary')
