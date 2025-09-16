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
    
def _fetch_news_headlines(company_name):
    """Fetches news headlines using Google search."""
    logging.info(f"Fetching news for {company_name}...")
    headlines = []
    try:
        queries = [
            f'"{company_name}" earnings report',
            f'"{company_name}" new product OR leadership change OR major deal',
            f'"{company_name}" regulatory news OR legal issues'
        ]
        for query in queries:
            # Manually limit to 5 results per query to avoid using unsupported arguments
            results_count = 0
            for url in search(query):
                if results_count >= 5:
                    break
                headline = url.split('/')[-1].replace('-', ' ').replace('_', ' ').title()
                if headline not in headlines:
                        headlines.append(headline)
                results_count += 1
            time.sleep(2) # Add a manual pause to be respectful to Google's servers
            if len(headlines) >= 10: break
        return headlines
    except Exception as e:
        logging.error(f"An error occurred while fetching news: {e}")
        return []
    
def _analyze_news():
    score=0
    """Analyzes the sentiment of recent news headlines for the stock."""
    company_name = info.get('longName', ticker_symbol)
    headlines = _fetch_news_headlines(company_name)
    analysis['news_headlines'] = headlines 

    if not headlines:
        analysis['news_sentiment'] = "No recent news found."
        analysis['news_summary'] = "Could not generate a summary as no news was found."
        score_breakdown['news_score'] = 0
        return

    avg_sentiment = sum(vader.polarity_scores(h)['compound'] for h in headlines) / len(headlines)

    if avg_sentiment > 0.5: sentiment_text, news_score = "Very Positive", 2
    elif avg_sentiment > 0.05: sentiment_text, news_score = "Positive", 1
    elif avg_sentiment < -0.5: sentiment_text, news_score = "Very Negative", -2
    elif avg_sentiment < -0.05: sentiment_text, news_score = "Negative", -1
    else: sentiment_text, news_score = "Neutral", 0
    
    analysis['news_sentiment'] = f"Overall sentiment is {sentiment_text} (Score: {avg_sentiment:.2f})"
    
    if groq_client:
        try:
            logging.info("Generating AI news summary with Groq...")
            prompt = f"Summarize the following news headlines for {company_name} in 2-3 key bullet points. Infer the overall sentiment (Positive, Negative, Neutral).\n\nHeadlines:\n" + "\n".join(headlines)
            chat_completion = groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.1-8b-instant")
            analysis['news_summary'] = chat_completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Failed to generate Groq summary: {e}")
            analysis['news_summary'] = "AI summary could not be generated."
    else:
        analysis['news_summary'] = "Groq API key not configured. AI summary unavailable."

    score_breakdown['news_score'] = news_score
    score += news_score

analysis,score_breakdown,score = {},{},0
_analyze_news()