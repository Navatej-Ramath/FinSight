import streamlit as st
import yfinance as yf
import pandas as pd
import logging
import os
import nltk
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from groq import Groq
from googlesearch import search

# --- NLTK Data Download ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    logging.info("Downloading the VADER sentiment lexicon for NLTK...")
    nltk.download('vader_lexicon')
    logging.info("Download complete.")

# --- Analyzer Class ---

class SwingAnalyzer:
    """
    Performs a comprehensive pre-market swing trading analysis based on technical,
    market, and news-based factors without external TA libraries.
    """
    def __init__(self, ticker_symbol):
        """Initializes the analyzer with a specific stock ticker."""
        self.ticker_symbol = ticker_symbol.upper()
        self.ticker = yf.Ticker(self.ticker_symbol)
        self.data = None
        self.analysis = {}
        self.score = 0
        self.score_breakdown = {}
        self.info = self.ticker.info
        self.vader = SentimentIntensityAnalyzer()
        self.sector_map = {
            'Financial Services': ('NIFTY BANK', '^NSEBANK'),
            'Technology': ('NIFTY IT', '^CNXIT'),
            'Healthcare': ('NIFTY PHARMA', '^CNXPHARMA'),
            'Automotive': ('NIFTY AUTO', '^CNXAUTO'),
            'Energy': ('NIFTY ENERGY', '^CNXENERGY'),
            'Consumer Durables': ('NIFTY CONSUMER DURABLES', '^CNXCONSUM'),
            'Fast Moving Consumer Goods': ('NIFTY FMCG', '^CNXFMCG'),
            'Metals & Mining': ('NIFTY METAL', '^CNXMETAL')
        }
        try:
            if os.getenv("GROQ_API_KEY"):
                self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            else:
                self.groq_client = None
        except Exception as e:
            logging.error(f"Failed to initialize Groq client: {e}")
            self.groq_client = None
    
    def _fetch_data(self):
        """Fetches historical data sufficient for analysis (e.g., 1 year for 200-day SMA)."""
        try:
            self.data = self.ticker.history(period="1y", interval="1d")
            if self.data.empty:
                logging.error(f"No daily data found for {self.ticker_symbol}")
                return False
            self.data = self.data[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
            return True
        except Exception as e:
            logging.error(f"Error fetching data for {self.ticker_symbol}: {e}")
            return False

    def _calculate_rsi(self, data, period=14):
        """Calculate the Relative Strength Index (RSI) manually."""
        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).ewm(com=period-1, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(com=period-1, adjust=False).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _detect_candlestick(self, data):
        """
        Detects common candlestick patterns IN THE CONTEXT of a prior trend.
        A pattern is only valid if it appears after a move to reverse.
        """
        if len(data) < 21: return "Not enough data", 0 # Need 20 periods for SMA context

        # --- 1. Establish Trend Context ---
        # We'll use a 20-period Simple Moving Average to define the short-term trend.
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        
        last = data.iloc[-1]
        prev = data.iloc[-2]
        # Check if the price was recently trending down or up before the pattern formed.
        # We check the close of the *previous* candle against the moving average.
        was_in_downtrend = prev['Close'] < data['SMA20'].iloc[-2]
        was_in_uptrend = prev['Close'] > data['SMA20'].iloc[-2]

        body_size = abs(last['Close'] - last['Open'])
        
        # --- 2. Look for Bullish Reversal Patterns (only if in a downtrend) ---
        if was_in_downtrend:
            # Bullish Engulfing: A strong green candle that "engulfs" the prior red candle.
            if (last['Close'] > last['Open'] and prev['Close'] < prev['Open'] and # Green engulfs red
                last['Close'] > prev['Open'] and last['Open'] < prev['Close']):
                return "Bullish Engulfing (Reversal) pattern detected", 15 # Strong signal

            # Hammer: A small body at the top with a long lower wick, showing buyers stepped in.
            lower_wick = last['Open'] - last['Low'] if last['Open'] < last['Close'] else last['Close'] - last['Low']
            upper_wick = last['High'] - last['Close'] if last['Open'] < last['Close'] else last['High'] - last['Open']
            if body_size > 0 and lower_wick > body_size * 2 and upper_wick < body_size * 0.5:
                return "Hammer (Reversal) pattern detected", 15 # Strong signal

        # --- 3. Look for Bearish Reversal Patterns (only if in an uptrend) ---
        if was_in_uptrend:
            # Bearish Engulfing: A strong red candle that "engulfs" the prior green candle.
            if (last['Close'] < last['Open'] and prev['Close'] > prev['Open'] and # Red engulfs green
                last['Open'] > prev['Close'] and last['Close'] < prev['Open']):
                return "Bearish Engulfing (Reversal) pattern detected", -15 # Strong signal

            # Shooting Star: A small body at the bottom with a long upper wick, showing sellers took control.
            lower_wick = last['Open'] - last['Low'] if last['Open'] < last['Close'] else last['Close'] - last['Low']
            upper_wick = last['High'] - last['Close'] if last['Open'] < last['Close'] else last['High'] - last['Open']
            if body_size > 0 and upper_wick > body_size * 2 and lower_wick < body_size * 0.5:
                return "Shooting Star (Reversal) pattern detected", -15 # Strong signal
        
        # --- 4. Check for Indecision ---
        range_size = last['High'] - last['Low']
        if body_size / range_size < 0.1 if range_size > 0 else True:
            return "Doji/Indecision pattern detected", 0

        # --- 5. If no specific pattern, just analyze the candle itself ---
        if last['Close'] > last['Open']:
            return "Bullish Candle detected", 7
        else:
            return "Bearish Candle detected", -7

    def _analyze_index(self, ticker):
        """Helper function to analyze the trend and momentum of any index."""
        
        try:
            curr_trend_score=0
            curr_trend_text=""
            data = yf.Ticker(ticker).history(period="50d", interval="1d")
            if len(data) < 21: return 0 # Not enough data to analyze

            # Use a 20-day SMA to determine the short-term trend
            data['SMA20'] = data['Close'].rolling(window=20).mean()
            latest = data.iloc[-1]
            sma20 = data['SMA20'].iloc[-1]
            
            # Determine trend direction
            is_uptrend = latest['Close'] > sma20
            
            # Check the strength of the last candle
            is_strong_green = latest['Close'] > latest['Open'] and (latest['Close'] - latest['Open']) / (latest['High'] - latest['Low']) > 0.6 if (latest['High'] - latest['Low']) > 0 else False

            is_strong_red = latest['Open'] > latest['Close'] and (latest['Open'] - latest['Close']) / (latest['High'] - latest['Low']) > 0.6 if (latest['High'] - latest['Low']) > 0 else False
            
            # Score based on trend and momentum
            if is_uptrend:
                if is_strong_green:
                    curr_trend_text="Bullish"
                    curr_trend_score=2
                else:
                    curr_trend_text="Neutral"
                    curr_trend_score=1
                                
            else:
                if is_strong_red:
                    curr_trend_text="Bearish"
                    curr_trend_score=-2
                else:
                    curr_trend_text="Neutral"
                    curr_trend_score=-1
        except Exception as e:
            logging.error(f"Could not analyze index {ticker}: {e}")# Return neutral score if index fetch fails
        return curr_trend_score, curr_trend_text

    def _analyze_technicals(self):
        """A. Perform technical analysis on the stock chart."""
        
        #Trend Analysis using 50-day and 200-day SMAs.
        
        
        if len(self.data) < 200:
            self.analysis['technical_summary'] = "Not enough data for full technical analysis."
            return
        self.data['SMA50'] = self.data['Close'].rolling(window=50).mean()
        self.data['SMA200'] = self.data['Close'].rolling(window=200).mean()
        latest_close = self.data['Close'].iloc[-1]
        latest_sma50 = self.data['SMA50'].iloc[-1]
        latest_sma200 = self.data['SMA200'].iloc[-1]
        trend_score = 0
        if latest_close > latest_sma50 > latest_sma200:
            trend_text = "Clear Uptrend: Price > 50-day SMA > 200-day SMA."
            trend_score = 10
        elif latest_close < latest_sma50 < latest_sma200:
            trend_text = "Clear Downtrend: Price < 50-day SMA < 200-day SMA."
            trend_score = -10
        elif latest_close > latest_sma50 and latest_close > latest_sma200:
            trend_text = "Bullish: Price is above key moving averages."
            trend_score = 5
        else:
            trend_text = "Bearish: Price is below key moving averages." if latest_close < latest_sma50 else "Sideways Trend:price is between moving averages."
            trend_score = -5 if latest_close < latest_sma50 else 0
        self.analysis['trend'] = trend_text
        self.score_breakdown['trend_score'] = trend_score
        self.score += trend_score


        # Support and Resistance Analysis


        last_10_days = self.data.tail(10)
        current_close = self.data['Close'].iloc[-1]       
        previous_close = self.data['Close'].iloc[-2]
        immediate_support = last_10_days['Low'].min()
        immediate_resistance = last_10_days['High'].max()
        support_distance_pct = (current_close - immediate_support) / current_close * 100
        resistance_distance_pct = (immediate_resistance - current_close) / current_close * 100
        key_score = 0
        if (current_close > previous_close and  # Green candle
            current_close > immediate_support and  # Above support
            support_distance_pct < 2.0):  # Close to support (within 2%)
            self.analysis["support/resis"] = "BULLISH - Price holding above support with upward momentum"
            key_score = 8

        elif (current_close < previous_close and  # Red candle
            current_close < immediate_resistance and  # Below resistance
            resistance_distance_pct < 2.0):  # Close to resistance (within 2%)
            self.analysis["support/resis"] = "BEARISH - Price rejected at resistance with downward momentum"
            key_score = 8

        elif current_close > immediate_resistance:
            self.analysis["support/resis"] = "BULLISH BREAKOUT - Price broke above resistance"
            key_score = 15

        elif current_close < immediate_support:
            self.analysis["support/resis"] = "BEARISH BREAKDOWN - Price broke below support"
            key_score = -15

        elif support_distance_pct < resistance_distance_pct:
            self.analysis["support/resis"] = "NEUTRAL-BULLISH - Closer to support than resistance"
            key_score = 4  # Slightly positive for being closer to support

        else:
            self.analysis["support/resis"] = "NEUTRAL-BEARISH - Closer to resistance than support"
            key_score = -4  # Slightly negative for being closer to resistance
        self.score_breakdown['key_level_score'] = key_score
        self.score += key_score
        self.analysis['support_level'] = f"${immediate_support:.2f}"
        self.analysis['resistance_level'] = f"${immediate_resistance:.2f}"
        self.analysis['distance_to_support'] = f"{support_distance_pct:.1f}%"
        self.analysis['distance_to_resistance'] = f"{resistance_distance_pct:.1f}%"
              
        # Volume Analysis
        
        
        self.data['Avg_Volume_20'] = self.data['Volume'].rolling(window=20).mean()
        latest_volume = self.data['Volume'].iloc[-1]
        avg_volume = self.data['Avg_Volume_20'].iloc[-1]
        price_change = self.data['Close'].diff().iloc[-1]
        volume_ratio = latest_volume / avg_volume
        self.analysis['price_change'] = round(price_change,2)
        volume_score = 0
        if volume_ratio > 2.0:  # Extremely high volume
            if price_change > 0:
                volume_text = "EXTREMELY High Volume on UP day (Very strong buying)"
                volume_score = 10
            else:
                volume_text = "EXTREMELY High Volume on DOWN day (Very strong selling)"
                volume_score = -10
                
        elif volume_ratio > 1.5:  # Very high volume
            if price_change > 0:
                volume_text = "Very High Volume on UP day (Strong buying)"
                volume_score = 8
            else:
                volume_text = "Very High Volume on DOWN day (Strong selling)"
                volume_score = -8
                
        elif volume_ratio > 1.2:  # High volume
            if price_change > 0:
                volume_text = "High Volume on UP day (Good buying interest)"
                volume_score = 5
            else:
                volume_text = "High Volume on DOWN day (Good selling pressure)"
                volume_score = -5
                
        elif volume_ratio < 0.7:  # Low volume
            volume_text = "Low Volume (Lack of conviction)"
            volume_score = -5 if price_change < 0 else 0
            
        else:  # Normal volume
            if price_change > 0:
                volume_text = "Normal Volume on UP day (Steady buying)"
                volume_score = 5
            else:
                volume_text = "Normal Volume on DOWN day (Steady selling)"
                volume_score = -5
        volume_text += f" ({volume_ratio:.1f}x average)"
        self.analysis['volume'] = volume_text
        self.analysis['volume_ratio'] = f"{volume_ratio:.1f}x"
        self.score_breakdown['volume_score'] = volume_score
        self.score += volume_score
        
        
        # Momentum Indicators: RSI and MACD
        
        
        """
        Calculates and scores momentum indicators like RSI and MACD with trend context.
        """
        # --- Trend Context using 50-day EMA ---
        self.data['EMA50'] = self.data['Close'].ewm(span=50, adjust=False).mean()
        latest_price = self.data['Close'].iloc[-1]
        latest_ema50 = self.data['EMA50'].iloc[-1]
        
        is_uptrend = latest_price > latest_ema50

        # --- RSI Analysis (Context-Aware) ---
        self.data['RSI'] = self._calculate_rsi(self.data) # Assuming you have a _calculate_rsi method
        latest_rsi = self.data['RSI'].iloc[-1]
        rsi_score = 0
        
        if is_uptrend:
            # In an uptrend, high RSI is a sign of STRENGTH (Momentum).
            if latest_rsi > 70:
                rsi_text = "hence the stock has very strong bullish momentum"
                rsi_score = 7
            elif latest_rsi > 50:
                rsi_text = "hence the stock has healthy bullish momentum"
                rsi_score = 3
            else: # RSI is falling below the midpoint, showing weakening momentum.
                rsi_text = "hence the stock has weakening momentum / dip"
                rsi_score = 0 # Neutral, as it's a "watch" signal for a potential bounce or further fall.
        
        else: # In a downtrend or sideways market, we look for reversals (Mean Reversion).
            if latest_rsi < 30:
                rsi_text = "hence the stock is oversold (potential bounce)"
                rsi_score = 7
            elif latest_rsi < 40:
                rsi_text = "Hence, the stock is approaching oversold levels"
                rsi_score = 3
            elif latest_rsi > 70:
                rsi_text = "hence the stock is overbought (potential drop)"
                rsi_score = -7
            elif latest_rsi > 60:
                rsi_text = "hence the stock is approaching overbought levels"
                rsi_score = -3
            else: # Between 40-60, the stock is truly neutral.
                rsi_text = "hence the stock is neutral"
                rsi_score = 0
        self.score_breakdown['rsi_score'] = rsi_score
        self.analysis['rsi'] = f"RSI Score is {latest_rsi:.2f} {rsi_text}"
        self.score += rsi_score # You would add this to your overall score later

        # --- MACD Analysis (Coming next) ---
        # We will add the MACD logic here in the next step.
        
        ema12 = self.data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD_line'] = ema12 - ema26
        self.data['Signal_line'] = self.data['MACD_line'].ewm(span=9, adjust=False).mean()
        
        latest_macd, prev_macd = self.data['MACD_line'].iloc[-1], self.data['MACD_line'].iloc[-2]
        latest_signal, prev_signal = self.data['Signal_line'].iloc[-1], self.data['Signal_line'].iloc[-2]
        
        macd_score = 0
        # A bullish crossover is the strongest signal, warranting a +2 score.
        if latest_macd > latest_signal and prev_macd <= prev_signal:
            macd_text, macd_score = "MACD line crossed above Signal line - Bullish crossover signal", 8
        elif latest_macd < latest_signal and prev_macd >= prev_signal:
            macd_text, macd_score = "MACD line crossed below Signal line - Bearish crossover signal", -8
        elif latest_macd > latest_signal:
            macd_text, macd_score = "MACD remains above Signal line - Bullish momentum continues", 4
        else: # latest_macd < latest_signal
            macd_text, macd_score = "MACD remains below Signal line - Bearish momentum continues", -4
                    
        self.analysis['macd'] = macd_text        
        self.score_breakdown['macd_score'] = macd_score
        
        #detect simple candlestick patterns
        
        
        pattern_text, pattern_score = self._detect_candlestick(self.data)
        self.analysis['candlestick'] = pattern_text
        self.score_breakdown['candlestick_score'] = pattern_score
        self.score += pattern_score
    
    def _analyze_market_sentiment(self):
        """
        Analyzes the broader market sentiment by checking the NIFTY 50 and SENSEX indices.
        Scores the market from -2 (Strongly Bearish) to +2 (Strongly Bullish).
        """
        
        
        # analyze market sentiment
        
        
        logging.info("Analyzing broader market sentiment...")
        index_tickers = {'NIFTY 50': '^NSEI', 'SENSEX': '^BSESN'}
        total_score=0
        for name, ticker in index_tickers.items():
            score, trend_text = self._analyze_index(ticker)
            total_score += score
            self.analysis[f"{name}_trend"] = trend_text
        indian_market_final_score=0
        final_score = round(total_score / len(index_tickers)) if index_tickers else 0
        if final_score >= 2: market_sentiment_text,indian_market_final_score = "The overall market is Strongly Bullish",10
        elif final_score == 1: market_sentiment_text,indian_market_final_score = "The overall market is Mildly Bullish",5
        elif final_score == -1: market_sentiment_text,indian_market_final_score = "The overall market is Mildly Bearish",-5
        elif final_score <= -2: market_sentiment_text,indian_market_final_score = "The overall market is Strongly Bearish",-10
        else: market_sentiment_text,indian_market_final_score = "The overall market is Neutral / Mixed",0
            
        self.score_breakdown['market_sentiment_score'] = indian_market_final_score
        self.analysis['market_sentiment_text'] = market_sentiment_text
        self.score+=indian_market_final_score
        
        
        # analyse the current sector sentiment

        
        """Analyzes the stock's specific sector index."""
        logging.info("Analyzing sector sentiment...")
        sector = self.info.get('sector')
        sector_final_score=0
        if sector in self.sector_map:
            sector_name, sector_ticker = self.sector_map[sector]
            score,trend_text = self._analyze_index(sector_ticker)
            if score >= 2:
                sector_sentiment_text,sector_final_score = f"Strong bullish trend confirmed in {sector_name} sector",15
            elif score == 1:
                sector_sentiment_text,sector_final_score = f"Slight bullish bias detected in {sector_name} sector",8
            elif score == -1:
                sector_sentiment_text,sector_final_score = f"Moderate bearish pressure in {sector_name} sector",-8
            elif score <= -2:
                sector_sentiment_text,sector_final_score = f"Intense bearish momentum across {sector_name} sector",-15
            else:
                sector_sentiment_text,sector_final_score = f"Neutral stance: {sector_name} sector lacks clear direction",0
            self.score_breakdown['sector_sentiment_score'] = sector_final_score
            self.analysis['sector_sentiment_text'] = sector_sentiment_text
        else:
            self.score_breakdown['sector_sentiment_score'] = 0
            self.analysis['sector_sentiment_text'] = "Sector analysis not available."
        self.score+=self.score_breakdown['sector_sentiment_score']


        # analyze global market sentiment
        
        
        """
        Analyzes global cues from US (previous close) and Asian markets.
        """
        logging.info("Analyzing global market sentiment...")
        us_tickers = {'S&P 500': '^GSPC', 'NASDAQ': '^IXIC'}
        asia_tickers = {'Nikkei 225': '^N225', 'Hang Seng': '^HSI'}      
        total_score = 0
        global_final_score=0
        num_indices = len(us_tickers) + len(asia_tickers)
        for name, ticker in (us_tickers.items() | asia_tickers.items()):
            score, global_trend_text = self._analyze_index(ticker) # We only need the score for aggregation
            total_score += score
            self.analysis[f"{name}_trend"] = global_trend_text
        final_score = round(total_score / num_indices) if num_indices > 0 else 0
        if final_score >= 2:
            global_sentiment_text,global_final_score = "Strongly positive cues from global markets.",5
        elif final_score == 1:
            global_sentiment_text,global_final_score = "Mildly positive sentiment from global markets.",2
        elif final_score == -1:
            global_sentiment_text,global_final_score = "Mildly negative sentiment from global markets.",-2
        elif final_score <= -2:
            global_sentiment_text,global_final_score = "Strongly negative cues from global markets.",-5
        else:
            global_sentiment_text,global_final_score = "Neutral or mixed signals from global markets.",0
        self.score_breakdown['global_sentiment_score'] = global_final_score
        self.analysis['global_sentiment_text'] = global_sentiment_text
        self.score += global_final_score
        
    def _fetch_headlines(self, queries, headline_limit):
        """Generic function to fetch news headlines for a list of queries."""
        headlines = []
        try:
            for query in queries:
                count = 0
                for url in search(query):
                    if count >= 7: break
                    headline = url.split('/')[-1].replace('-', ' ').replace('_', ' ').title()
                    if headline not in headlines:
                        headlines.append(headline)
                    count += 1
                time.sleep(1) # Pause between queries
                if len(headlines) >= headline_limit: break
            return headlines
        except Exception as e:
            logging.error(f"An error occurred while fetching news: {e}")
            return []

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

    def _analyze_macro_news(self):
        """Analyzes sentiment of macro-economic news."""
        queries = [
            '"India GDP" OR "inflation CPI" OR "IIP data" latest',
            '"RBI monetary policy" OR "repo rate" OR "interest rates" decision',
            '"India trade deficit" OR "exports imports" OR "FDI FII flows" data',
        ]
        headlines = self._fetch_headlines(queries, 15)
        self.analysis['macro_news_headlines'] = headlines
        if not headlines:
            self.analysis['macro_news_sentiment'] = "No recent macro news found."
            self.score_breakdown['macro_news_score'] = 0
            return

        avg_sentiment = sum(self.vader.polarity_scores(h)['compound'] for h in headlines) / len(headlines)
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
        self.analysis['macro_news_sentiment'] = f"Overall sentiment is {sentiment_text} (Score: {avg_sentiment:.2f})"
        self.score_breakdown['macro_news_score'] = news_score
        self.score += news_score

        prompt = "Summarize the following macro-economic news headlines for India in 2-3 key bullet points.\n\nHeadlines:\n" + "\n".join(headlines)
        self._groq_summary(prompt, 'macro_news_summary')

            
    def _generate_final_summary(self):
        """Calculates the final score, max score, and trading bias."""
        score = self.score
        
        # Define the maximum possible score for each component
        max_scores = {
            'trend_score': 10,
            'key_level_score': 15,
            'volume_score': 10,
            'rsi_score': 7,
            'macd_score': 8,
            'candlestick_score': 15,
            'market_sentiment_score': 10,
            'sector_sentiment_score': 15,
            'global_sentiment_score': 5,
            'company_news_score': 10, # Only if news is analyzed
            'sector_news_score': 5,    # Only if news is analyzed
            'macro_news_score': 5      # Only if news is analyzed
        }
        
        # Calculate max score based on which analyses were run
        max_score = sum(v for k, v in max_scores.items() if k in self.score_breakdown)
        
        self.analysis['max_score'] = max_score
        
        # Normalize score to a percentage for consistent bias calculation
        score_percent = (score / max_score) * 100 if max_score > 0 else 0

        if score_percent >= 60: bias = "Strong Bullish Bias"
        elif 30 <= score_percent < 60: bias = "Cautiously Bullish"
        elif -30 < score_percent < 30: bias = "Neutral / Sideways"
        elif -60 < score_percent <= -30: bias = "Cautiously Bearish"
        else: bias = "Strong Bearish Bias"
        self.analysis['final_bias'] = bias

    def _generate_final_ai_summary(self):
        """Generates the final AI trading plan based on all collected data."""
        
        # Prepare a string of all the analysis findings
        summary_points = "\n".join([f"- {key.replace('_', ' ').title()}: {value}" for key, value in self.analysis.items()])

        prompt = f"""
        Act as a professional trading analyst providing a final recommendation for {self.ticker_symbol}.

        Here is the complete analysis data:
        {summary_points}
        
        Based on the **ENTIRE** set of data, provide a final, actionable trading plan in the following format:

        **1. Overall Bias:** State the final bias (e.g., Bullish, Bearish, Neutral) and briefly explain the primary reasons based on the total score and key factors (e.g., "The bias is Bullish due to a strong technical uptrend and positive market sentiment, despite neutral news.").

        **2. Key Levels to Watch:**
           - **Entry Trigger:** What price level breaking would confirm the bullish view? (e.g., "A move above the resistance at {self.analysis.get('resistance_level', 'N/A')}...").
           - **Stop-Loss:** Where should a trader consider placing a stop-loss? (e.g., "...with a stop-loss below the support at {self.analysis.get('support_level', 'N/A')}.").

        **3. Trade Confidence:** Rate your confidence in this trading setup (e.g., High, Medium, Low) and provide a one-sentence justification.
        """
        self._groq_summary(prompt, 'final_ai_summary')

    def run_initial_analysis(self):
        """Runs only the fast, non-network-intensive analysis (technicals and indices)."""
        if self._fetch_data():
            self._analyze_technicals()
            self._analyze_market_sentiment()
            self._generate_final_summary()
            self._generate_final_ai_summary()# Generate an initial summary without news
            self.analysis['score'] = self.score
            self.analysis['score_breakdown'] = self.score_breakdown
            return self.analysis
        return None

    # --- NEW METHOD 2: Slow News Analysis ---
    def run_news_analysis(self, existing_analysis):
        """
        Runs only the slow news analysis and adds the results to an existing
        analysis dictionary.
        """
        # Restore state from the initial analysis
        self.analysis = existing_analysis
        self.score = existing_analysis.get('score', 0)
        self.score_breakdown = existing_analysis.get('score_breakdown', {})
        
        # Run the slow news functions
        self._analyze_company_news()
        self._analyze_sector_news()
        self._analyze_macro_news()

        # Regenerate the final summary with the new news scores included
        self._generate_final_summary()
        self._generate_final_ai_summary()
        self.analysis['score'] = self.score
        self.analysis['score_breakdown'] = self.score_breakdown
        return self.analysis