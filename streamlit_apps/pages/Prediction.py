import yfinance as yf
from groq import Groq
import os
import logging
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Configuration ---
# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# IMPORTANT: Set your Groq API key as an environment variable
# For local development: export GROQ_API_KEY='your_api_key_here'

# --- Manual Technical Indicator Functions (No pandas_ta needed) ---

def calculate_ema(data, length):
    """Calculates Exponential Moving Average."""
    return data.ewm(span=length, adjust=False).mean()

def calculate_sma(data, length):
    """Calculates Simple Moving Average."""
    return data.rolling(window=length).mean()

def calculate_rsi(data, length=14):
    """Calculates Relative Strength Index."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, slow=26, fast=12, signal=9):
    """Calculates Moving Average Convergence Divergence."""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(high, low, close, length=14):
    """Calculates Average True Range."""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    tr = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close}).max(axis=1)
    return calculate_ema(tr, length)

def calculate_vwap(price, volume):
    """Calculates Volume Weighted Average Price."""
    return (price * volume).cumsum() / volume.cumsum()


class StockAnalyzer:
    """
    A comprehensive stock analyzer that combines quantitative data from yfinance
    with qualitative analysis from the Groq LLM, designed for Streamlit integration.
    """
    def __init__(self, ticker_symbol):
        """
        Initializes the analyzer for a given stock ticker.
        """
        self.ticker_symbol = ticker_symbol.upper()
        self.ticker = yf.Ticker(self.ticker_symbol)
        self.scores = {}
        self.max_scores = {}
        self.analysis_results = {}
        self.recommendation = ""
        self.final_score_percentage = 0

        try:
            if not os.getenv("GROQ_API_KEY"):
                logging.error("GROQ_API_KEY environment variable not set.")
                self.groq_client = None
            else:
                self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                logging.info("Groq client initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Groq client: {e}")
            self.groq_client = None

    def _get_yfinance_data(self):
        """Fetches all necessary data from yfinance."""
        logging.info(f"Fetching data for {self.ticker_symbol} from yfinance...")
        try:
            self.info = self.ticker.info
            if not self.info or 'symbol' not in self.info:
                raise ValueError(f"Invalid ticker or no data found for {self.ticker_symbol}")

            self.financials = self.ticker.financials
            self.balance_sheet = self.ticker.balance_sheet
            self.cashflow = self.ticker.cashflow
            
            if self.financials.empty or self.balance_sheet.empty:
                raise ValueError(f"Financial statements are empty for {self.ticker_symbol}. It might be a non-equity asset or delisted.")

            logging.info("Successfully fetched yfinance data.")
            return True
        except Exception as e:
            st.error(f"Could not fetch financial data for '{self.ticker_symbol}'. It may be an invalid ticker, a non-equity asset (like an index), or data may be unavailable. Error: {e}")
            logging.error(f"Could not fetch data for {self.ticker_symbol}. Error: {e}")
            return False

    def _get_financial_metric(self, data_frame, possible_keys, default=0):
        """
        Safely retrieves a financial metric from a DataFrame by checking a list of possible keys.
        """
        if data_frame.empty:
            logging.warning(f"DataFrame is empty. Cannot find keys {possible_keys}.")
            return default
            
        for key in possible_keys:
            if key in data_frame.index:
                return data_frame.loc[key].iloc[0]
        logging.warning(f"Could not find any of the keys {possible_keys} in the provided data. Returning default value {default}.")
        return default

    def _calculate_and_score_metrics(self):
        """Calculates derived metrics and scores them based on predefined criteria."""
        logging.info("Calculating and scoring metrics...")

        net_income = self._get_financial_metric(self.financials, ['Net Income', 'NetIncome'])
        total_revenue = self._get_financial_metric(self.financials, ['Total Revenue', 'TotalRevenue'])
        total_equity = self._get_financial_metric(self.balance_sheet, ['Stockholder Equity', 'Total Stockholder Equity', 'Total Equity', 'Shareholder Equity', 'Equity'])

        self.max_scores['ROE'] = 10
        roe = (net_income / total_equity) * 100 if total_equity and total_equity != 0 else 0
        self.analysis_results['ROE'] = f"{roe:.2f}%"
        if roe > 20: self.scores['ROE'] = 10
        elif roe > 15: self.scores['ROE'] = 8
        elif roe > 10: self.scores['ROE'] = 6
        elif roe > 5: self.scores['ROE'] = 4
        else: self.scores['ROE'] = 1

        self.max_scores['Net Margin'] = 10
        net_margin = (net_income / total_revenue) * 100 if total_revenue and total_revenue != 0 else 0
        self.analysis_results['Net Margin'] = f"{net_margin:.2f}%"
        if net_margin > 20: self.scores['Net Margin'] = 10
        elif net_margin > 10: self.scores['Net Margin'] = 8
        elif net_margin > 5: self.scores['Net Margin'] = 6
        else: self.scores['Net Margin'] = 3

        self.max_scores['Debt/Equity'] = 10
        total_debt = self.info.get('totalDebt', 0)
        d_e_ratio = total_debt / total_equity if total_equity and total_equity != 0 else float('inf')
        self.analysis_results['Debt/Equity'] = f"{d_e_ratio:.2f}"
        if d_e_ratio < 0.5: self.scores['Debt/Equity'] = 10
        elif d_e_ratio < 1.0: self.scores['Debt/Equity'] = 7
        elif d_e_ratio < 1.5: self.scores['Debt/Equity'] = 4
        else: self.scores['Debt/Equity'] = 1

        self.max_scores['P/E Ratio'] = 10
        pe_ratio = self.info.get('trailingPE')
        self.analysis_results['P/E Ratio'] = f"{pe_ratio:.2f}" if pe_ratio else "N/A"
        if pe_ratio:
            if pe_ratio < 15: self.scores['P/E Ratio'] = 10
            elif pe_ratio < 25: self.scores['P/E Ratio'] = 7
            elif pe_ratio < 40: self.scores['P/E Ratio'] = 4
            else: self.scores['P/E Ratio'] = 1
        else: self.scores['P/E Ratio'] = 0

        self.max_scores['PEG Ratio'] = 10
        peg_ratio = self.info.get('pegRatio')
        self.analysis_results['PEG Ratio'] = f"{peg_ratio:.2f}" if peg_ratio else "N/A"
        if peg_ratio:
            if peg_ratio < 1: self.scores['PEG Ratio'] = 10
            elif peg_ratio < 2: self.scores['PEG Ratio'] = 7
            else: self.scores['PEG Ratio'] = 2
        else: self.scores['PEG Ratio'] = 0
            
        self.max_scores['Payout Ratio'] = 5
        payout_ratio = self.info.get('payoutRatio', 0)
        self.analysis_results['Payout Ratio'] = f"{payout_ratio*100:.2f}%" if payout_ratio else "N/A"
        if 0 < payout_ratio < 0.6: self.scores['Payout Ratio'] = 5
        elif payout_ratio > 0.6: self.scores['Payout Ratio'] = 2
        else: self.scores['Payout Ratio'] = 0

        logging.info("Metrics calculated and scored.")

    def _get_qualitative_analysis(self):
        if not self.groq_client:
            logging.warning("Groq client not available or API key not set. Skipping qualitative analysis.")
            self.analysis_results['Qualitative Analysis'] = "Groq client not initialized. Please set the GROQ_API_KEY environment variable."
            return

        logging.info("Generating qualitative analysis with Groq...")
        company_name = self.info.get('longName', self.ticker_symbol)
        prompt = f"""
        Analyze **{company_name} ({self.ticker_symbol})** for a long-term investor. Provide a concise, expert-level qualitative analysis covering these areas with clear headings:
        **1. Competitive Advantage (Economic Moat):** What are its primary sustainable advantages (e.g., brand, network effects, switching costs)? How durable is its moat?
        **2. Management Quality & Corporate Governance:** Assess the CEO and leadership's track record and capital allocation strategy. Are there any governance concerns?
        **3. Industry Analysis & Growth Prospects:** Describe the industry's health and growth trends. What is the company's market position, and what are its key growth drivers and risks?
        Conclude with a summary of the key qualitative risks and opportunities.
        """
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
            )
            self.analysis_results['Qualitative Analysis'] = chat_completion.choices[0].message.content
            logging.info("Successfully generated qualitative analysis.")
        except Exception as e:
            logging.error(f"Failed to get qualitative analysis from Groq: {e}")
            self.analysis_results['Qualitative Analysis'] = f"Error fetching analysis from Groq: {e}"

    def _generate_final_verdict(self):
        total_score = sum(self.scores.values())
        max_score = sum(self.max_scores.values())
        self.final_score_percentage = (total_score / max_score) * 100 if max_score > 0 else 0

        if self.final_score_percentage >= 80:
            self.recommendation = "EXCELLENT candidate for long-term investment. Strong fundamentals and a positive outlook."
        elif self.final_score_percentage >= 60:
            self.recommendation = "GOOD candidate for long-term investment. Solid fundamentals, but some areas warrant closer inspection."
        elif self.final_score_percentage >= 40:
            self.recommendation = "NEUTRAL. The company has mixed fundamentals. Further research is essential before investing."
        else:
            self.recommendation = "CAUTION ADVISED. Significant weaknesses in fundamentals. May not be suitable for a conservative long-term investor."

    def run_analysis(self):
        if self._get_yfinance_data():
            self._calculate_and_score_metrics()
            self._get_qualitative_analysis()
            self._generate_final_verdict()
            return {
                "info": self.info,
                "quantitative": self.analysis_results,
                "scores": self.scores,
                "max_scores": self.max_scores,
                "qualitative": self.analysis_results.get('Qualitative Analysis'),
                "final_score": self.final_score_percentage,
                "recommendation": self.recommendation
            }
        return None

# --- Helper function to standardize column names ---
def standardize_columns(df):
    """Converts all DataFrame columns to lowercase to prevent KeyErrors."""
    df.columns = [col.lower() for col in df.columns]
    return df

# --- Streamlit Integration Functions ---

def short_term_analysis(ticker):
    """
    Runs a detailed short-term analysis (days to 2 years) and renders it in Streamlit.
    """
    st.header(f"Short-Term Analysis: {ticker}", divider="rainbow")
    with st.spinner(f"Running short-term analysis for {ticker}..."):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            df = stock.history(period="2y") # auto_adjust=False to get consistent column names
            if df.empty:
                st.error("Could not download historical data for the ticker.")
                return
            
            # Standardize column names to prevent KeyErrors
            df = standardize_columns(df)

            # Manual Indicator Calculations
            df['ema_50'] = calculate_ema(df['close'], 50)
            df['ema_200'] = calculate_ema(df['close'], 200)
            df['macd_line'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
            df['rsi_14'] = calculate_rsi(df['close'], 14)
            df['sma_20_vol'] = calculate_sma(df['volume'], 20)

            st.subheader("📊 Quantitative Factors")
            q_cols = st.columns(3)
            
            with q_cols[0]:
                st.markdown("**Trend Strength**")
                price = df['close'].iloc[-1]
                ema50 = df['ema_50'].iloc[-1]
                is_uptrend = price > ema50
                st.metric("Price vs 50 EMA", f"{'Above' if is_uptrend else 'Below'}", f"{'+' if is_uptrend else '-'} Signal")
                
                macd_hist = df['macd_hist']
                macd_rising = macd_hist.iloc[-1] > macd_hist.iloc[-2]
                st.metric("MACD Histogram", f"{'Rising' if macd_rising else 'Falling'}", f"{'+' if macd_rising else '-'} Signal")

                rsi = df['rsi_14'].iloc[-1]
                rsi_good = 40 <= rsi <= 80
                st.metric("RSI (14)", f"{rsi:.2f}", "In range" if rsi_good else "Extreme", delta_color="off")

            with q_cols[1]:
                st.markdown("**Volume & Valuation**")
                vol = df['volume'].iloc[-1]
                vol_sma20 = df['sma_20_vol'].iloc[-1]
                vol_strong = vol > vol_sma20
                st.metric("Volume vs 20-day Avg", f"{'Above' if vol_strong else 'Below'}", f"{'+' if vol_strong else '-'} Signal")
                
                peg = info.get('pegRatio', None)
                peg_good = peg is not None and peg < 1.2
                st.metric("PEG Ratio", f"{peg:.2f}" if peg else "N/A", "Good" if peg_good else "High")
                st.info("Delivery % data not available via yfinance.")

            with q_cols[2]:
                st.markdown("**Chart Pattern**")
                recent_60d = df.tail(60)
                last_high = recent_60d['high'].iloc[-1]
                prev_high = recent_60d.head(50)['high'].max()
                higher_high = last_high > prev_high
                st.metric("Higher Highs?", "Yes" if higher_high else "No", f"{'+' if higher_high else '-'} Signal")
                
                last_low = recent_60d['low'].iloc[-1]
                prev_low = recent_60d.head(50)['low'].min()
                higher_low = last_low > prev_low
                st.metric("Higher Lows?", "Yes" if higher_low else "No", f"{'+' if higher_low else '-'} Signal")

            st.subheader("📝 Qualitative Factors")
            qual_cols = st.columns(2)
            with qual_cols[0]:
                st.markdown("**Sector & Triggers**")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.info("Sector momentum & FII/DII flow data require specialized APIs.")
                try:
                    earnings_date = stock.calendar.iloc[0,0]
                    st.write(f"**Next Earnings:** {earnings_date.strftime('%Y-%m-%d') if isinstance(earnings_date, pd.Timestamp) else 'N/A'}")
                except Exception:
                    st.write("**Next Earnings:** Date not available.")

            with qual_cols[1]:
                st.markdown("**Liquidity & Red Flags**")
                st.info("Options OI & Block Deal data require specialized APIs.")
                with st.expander("🚨 View Potential Red Flags"):
                    if not is_uptrend: st.warning("Price is below its 50-day EMA, indicating a weak trend.")
                    if rsi > 75: st.warning(f"RSI is high at {rsi:.2f}, stock may be overbought.")
                    if not vol_strong: st.warning("Recent volume is below the 20-day average.")
                    if not (higher_high and higher_low): st.warning("The stock is not consistently making higher highs and higher lows.")

        except Exception as e:
            st.error(f"An error occurred during short-term analysis: {e}")
            logging.error(f"Short-term analysis failed for {ticker}: {e}")

def intraday_analysis(ticker):
    """
    Runs an end-of-day scan for next-day intraday trading setups.
    """
    st.header(f"Intraday Setup Analysis: {ticker}", divider="rainbow")
    st.caption("This is an end-of-day scan to identify potential setups for the *next* trading day.")
    with st.spinner(f"Running intraday setup scan for {ticker}..."):
        try:
            stock = yf.Ticker(ticker)
            df_daily = stock.history(period="1mo", interval="1d")
            df_intraday = stock.history(period="2d", interval="5m")
            if df_intraday.empty or df_daily.empty:
                st.error("Could not download intraday data.")
                return

            # Standardize column names to prevent KeyErrors
            df_daily = standardize_columns(df_daily)
            df_intraday = standardize_columns(df_intraday)

            # Manual Indicator Calculations
            df_daily['atr_14'] = calculate_atr(df_daily['high'], df_daily['low'], df_daily['close'], 14)
            df_intraday['vwap_d'] = df_intraday.groupby(df_intraday.index.date, group_keys=False).apply(lambda x: calculate_vwap(x['close'], x['volume']))
            df_intraday['sma_10'] = calculate_sma(df_intraday['close'], 10)
            df_intraday['sma_20'] = calculate_sma(df_intraday['close'], 20)

            st.subheader("📊 Quantitative Setup (from previous day)")
            q_cols = st.columns(3)
            last_day_intra = df_intraday.loc[df_intraday.index.date == df_intraday.index.date[-1]]
            
            with q_cols[0]:
                st.markdown("**Key Levels & Trend**")
                open_range_high = last_day_intra.head(6)['high'].max()
                open_range_low = last_day_intra.head(6)['low'].min()
                st.write(f"**Opening 30m Range:**\n- High: `{open_range_high:.2f}`\n- Low: `{open_range_low:.2f}`")
                
                trend_signal = df_intraday['sma_10'].iloc[-1] > df_intraday['sma_20'].iloc[-1]
                st.metric("Short-Term Trend (10/20 SMA)", "Uptrend" if trend_signal else "Downtrend", delta_color="off")

            with q_cols[1]:
                st.markdown("**Price vs VWAP**")
                last_price = last_day_intra['close'].iloc[-1]
                last_vwap = last_day_intra['vwap_d'].iloc[-1]
                above_vwap = last_price > last_vwap
                st.metric("End-of-Day Price vs VWAP", "Above" if above_vwap else "Below", f"{'+' if above_vwap else '-'} Signal")
                
            with q_cols[2]:
                st.markdown("**Volatility & F&O**")
                atr = df_daily['atr_14'].iloc[-2]
                atr_pct = (atr / df_daily['close'].iloc[-2]) * 100
                st.metric("ATR (Volatility)", f"{atr_pct:.2f}% of price", "High" if atr_pct > 3 else "Normal")
                st.info("Real-time Order Flow & Rollover % require specialized APIs.")

            st.subheader("📝 Qualitative Context")
            qual_cols = st.columns(2)
            with qual_cols[0]:
                st.markdown("**Global Market Clues**")
                nifty = yf.Ticker("^NSEI").history(period="1d")
                nifty = standardize_columns(nifty)
                nifty_change = nifty['close'].pct_change().iloc[-1] * 100
                st.metric("NIFTY 50 Change", f"{nifty_change:.2f}%", "Up" if nifty_change > 0 else "Down")

            with qual_cols[1]:
                st.markdown("**News & Red Flags**")
                st.info("Check for major pre-market news or results announcements manually.")
                with st.expander("🚨 View Potential Red Flags for Next Day"):
                    if not above_vwap: st.warning("Stock closed below VWAP, indicating weakness.")
                    if not trend_signal: st.warning("Short-term moving averages indicate a downtrend.")
                    if atr_pct < 1: st.warning("Low volatility (ATR) might lead to choppy movement.")

        except Exception as e:
            st.error(f"An error occurred during intraday analysis: {e}")
            logging.error(f"Intraday analysis failed for {ticker}: {e}")

def long_term_analysis(ticker):
    """
    Runs the long-term analysis and renders the results in Streamlit.
    """
    st.header(f"Long-Term Analysis: {ticker}", divider="rainbow")
    analyzer = StockAnalyzer(ticker)
    
    if not os.getenv("GROQ_API_KEY"):
        st.warning("`GROQ_API_KEY` environment variable not set. Qualitative analysis will be skipped.")

    with st.spinner(f"Running comprehensive analysis for {ticker}..."):
        results = analyzer.run_analysis()
        
    if results:
        st.subheader("📊 Quantitative Analysis & Scoring")
        cols = st.columns(3)
        metrics_to_display = ['ROE', 'Net Margin', 'Debt/Equity', 'P/E Ratio', 'PEG Ratio', 'Payout Ratio']
        for i, metric in enumerate(metrics_to_display):
            with cols[i % 3]:
                value = results["quantitative"].get(metric, "N/A")
                score = results["scores"].get(metric, 0)
                max_s = results["max_scores"].get(metric, 0)
                st.metric(label=f"{metric} (Score: {score}/{max_s})", value=value)

        st.subheader("🧭 Qualitative Analysis (via Groq LLM)")
        with st.expander("Click to read the detailed qualitative analysis"):
            st.markdown(results.get("qualitative", "Analysis not available."))

        st.subheader("🏆 Final Verdict")
        score_percentage = results.get('final_score', 0)
        st.progress(int(score_percentage), text=f"Quantitative Score: {score_percentage:.2f}%")
        st.write(results.get("recommendation", "Could not generate a recommendation."))
        st.caption("Disclaimer: This is an AI-generated analysis and not financial advice. Always do your own research.")

def render_prediction_page(ticker: str = "AAPL"):
    """
    Render the prediction page with stock data and AI insights.
    """
    st.title("🤖 AI-Powered Stock Insights")
    st.info("This version uses manual calculations for technical indicators to avoid library conflicts.")

    with st.sidebar:
        st.markdown("## 💼 Investment Type")
        investment_type = st.selectbox(
            "Select Investment Type",
            options=["Long-term", "Short-term", "Intraday"],
            index=0
        )
    
    if ticker:
        if investment_type == "Intraday":
            intraday_analysis(ticker)
        elif investment_type == "Short-term":
            short_term_analysis(ticker)
        elif investment_type == "Long-term":
            long_term_analysis(ticker)
    else:
        st.warning("Please enter a stock ticker to begin analysis.")

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    ticker_symbol_for_test = st.text_input("Enter a stock ticker for testing:", "MSFT").upper()
    render_prediction_page(ticker_symbol_for_test)
