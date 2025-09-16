import yfinance as yf
from groq import Groq
import os
import logging
import streamlit as st

from investment_type.long_term_invest import StockAnalyzer
from investment_type.short_term_swing import SwingAnalyzer
from utils.helpers import result_message, simple_trend_box, render_final_score_gauge
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def render_long_term_results(results):
    metric_info = {
        'ROE': {"explanation": "How well the company uses shareholder money to make a profit.", "preferred": "Preferred: > 15%"},
        'Net Margin': {"explanation": "The percentage of sales that is actual profit.", "preferred": "Preferred: > 10%"},
        'Debt/Equity': {"explanation": "Compares company debt to shareholder equity.", "preferred": "Preferred: < 1.0"},
        'Current Ratio': {"explanation": "Ability to pay short-term debts. (Assets / Liabilities)", "preferred": "Preferred: > 1.5"},
        'Quick Ratio': {"explanation": "A stricter look at short-term liquidity, excluding inventory.", "preferred": "Preferred: > 1.0"},
        'Asset Turnover': {"explanation": "How efficiently assets are used to generate sales.", "preferred": "Varies by industry"},
        'Inventory Turnover': {"explanation": "How quickly a company sells its inventory.", "preferred": "Varies by industry"},
        'Free Cash Flow': {"explanation": "Cash left after paying for operations and investments. Positive is essential.", "preferred": "Preferred: > 0"},
        'OCF / Sales': {"explanation": "The cash generated for every dollar of sales. Shows cash profitability.", "preferred": "Preferred: > 15%"},
        'P/E Ratio': {"explanation": "Stock price relative to its earnings per share.", "preferred": "Preferred: < 25"},
        'PEG Ratio': {"explanation": "P/E ratio relative to earnings growth.", "preferred": "Preferred: < 1.0"},
        'Payout Ratio': {"explanation": "Percentage of earnings paid to shareholders as dividends.", "preferred": "Preferred: < 60%"}
    }
    metric_groups = {
        "Profitability Ratios": ['ROE', 'Net Margin'],
        "Liquidity Ratios": ['Current Ratio', 'Quick Ratio'],
        "Solvency Ratios": ['Debt/Equity'],
        "Cash Flow Ratios": ['Free Cash Flow', 'OCF / Sales'], # <-- NEW CATEGORY
        "Efficiency Ratios": ['Asset Turnover', 'Inventory Turnover'],
        "Valuation & Payout": ['P/E Ratio', 'PEG Ratio', 'Payout Ratio']
    }
    for category, metrics in metric_groups.items():
        st.markdown(f"**{category}**")
        if category == "Liquidity Ratios" and "LiquidityWarning" in results["quantitative"]:
            st.warning(results["quantitative"]["LiquidityWarning"])
        if category == "Efficiency Ratios" and "InventoryTurnoverWarning" in results["quantitative"]:
            st.warning(results["quantitative"]["InventoryTurnoverWarning"])
            st.divider()
        visible_metrics = [m for m in metrics if results["quantitative"].get(m) != "N/A"]       
        if visible_metrics:
            cols = st.columns(len(visible_metrics))
            for i, metric in enumerate(visible_metrics):
                with cols[i]:
                    value = results["quantitative"].get(metric)
                    score = results["scores"].get(metric, 0)
                    max_s = results["max_scores"].get(metric, 0)                   
                    st.metric(label=f"{metric} (Score: {score}/{max_s})", value=value)
                    info = metric_info.get(metric)
                    if info:
                        st.markdown(f"<small>{info['explanation']}<br>{info['preferred']}</small>", unsafe_allow_html=True)       
        st.divider()
    st.subheader("🧭 Qualitative Analysis (AI-Generated)")
    with st.expander("Click to read the detailed qualitative analysis"):
        st.markdown(results.get("qualitative", "Analysis not available."))

    st.subheader("🏆 Final Verdict")
    score_percentage = results.get('final_score', 0)
    st.progress(int(score_percentage), text=f"Quantitative Score: {score_percentage:.2f}%")
    st.markdown(f"**Recommendation:** {results.get('recommendation', 'Could not generate a recommendation.')}")
    st.caption("Disclaimer: This is an AI-generated analysis and not financial advice.")
def long_term_analysis(ticker):
    """
    Runs the long-term analysis using the StockAnalyzer class and renders the results.
    """
    st.header(f"Long-Term Analysis: {ticker}", divider="rainbow")
    if not os.getenv("GROQ_API_KEY"):
        st.warning("`GROQ_API_KEY` environment variable not set. Qualitative analysis will be disabled.")

    with st.spinner(f"Running comprehensive analysis for {ticker}..."):
        analyzer = StockAnalyzer(ticker)
        results = analyzer.run_analysis()
        
        if results:
            render_long_term_results(results)
        else:
            st.warning("Analysis could not be completed.")
 
def render_short_term_swing_analysis(results,show_sentiment):
    """
    Renders the analysis results in a structured format in Streamlit.
    """ 
    if not results:
        st.error("Analysis could not be completed.")
        return
    st.subheader("Pre-Market Analysis")
       
    st.markdown("##### 1. Identify the Trend")
    st.caption("Is the stock in a clear uptrend, downtrend, or moving sideways on the daily chart? Use simple moving averages (e.g., 50-day and 200-day SMA). If the price is above these, the trend is generally bullish.")

    result_message( results.get('trend', 'N/A'), results.get('score_breakdown', {}).get('trend_score'),10)
    st.markdown("---")

    st.markdown("##### 2. Key Support and Resistance Levels")
    st.caption("For a swing trade, you want to buy near support and sell near resistance.")
    st.info(f"**Support:** {results.get('support_level', 'N/A')} (Potential buy zone)")
    st.info(f"**Resistance:** {results.get('resistance_level', 'N/A')} (Potential sell zone)")
    st.info(f"**Distance to Support Level:** {results.get('distance_to_support', 'N/A')}  and  **Distance to Resistance Level:** {results.get('distance_to_resistance', 'N/A')} ")

    # Safely get the key_level_score with default value 0
    key_level_score = results.get('score_breakdown', {}).get('key_level_score', 0)
    result_message(results.get('support/resis', 'No analysis available'), key_level_score,15)

    st.markdown("---")


    st.markdown("##### 3. Volume Analysis")
    st.caption("Was the most recent day's trading volume higher or lower than average? High volume on an up day can indicate strong buying interest; high volume on a down day can indicate strong selling pressure.")
    st.info(f"**Volume Ratio(Latest/Average Volume)** is {results.get('volume_ratio', 'N/A')} and **Price Change** is {results.get('price_change')} hence this indicates {'UP day' if (results.get('price_change') > 0) else 'Down day' }")
    volume_score = results.get('score_breakdown', {}).get('volume_score', 0)
    result_message(results.get('volume', 'No analysis available'), volume_score,10)
    st.markdown("---")

    st.markdown("##### 4. Momentum Indicators")
    st.caption("RSI: Is it above 70 (overbought) or below 30 (oversold)? MACD: Is the MACD line above the signal line (bullish) or below it (bearish)?")
    rsi_score = results.get('score_breakdown', {}).get('rsi_score', 0)
    result_message(results.get('rsi', 'No analysis available'), rsi_score,7)
    macd_score = results.get('score_breakdown', {}).get('macd_score', 0)
    result_message(results.get('macd', 'No analysis available'), macd_score,8)
    st.markdown("---")
    st.markdown("##### 5. Candlestick Patterns")
    st.caption("Patterns like 'Bullish Engulfing' or 'Hammer' can hint at a potential reversal to the upside, while patterns like 'Bearish Engulfing' or 'Shooting Star' can hint at a downside move.")
    result_message(results.get('candlestick', 'No analysis available'), results.get('score_breakdown', {}).get('candlestick_score', 0),15)
    st.markdown("***")
    st.subheader("Broader Market & Sector Sentiment")
       
    st.markdown("##### 1. Market Indices")
    st.caption("Check the trend of the major indices (NIFTY 50, SENSEX). Is the overall market bullish or bearish? A strong stock often can't fight a weak overall market.")
    # Create 3 columns and unpack into 3 variables
    col_left, col_center, col_right = st.columns([0.9, 1, 1])

    with col_left:
        col_inner1, col_inner2 = st.columns(2, gap="small") 
        # Add gap between boxes
        with col_inner1:
            simple_trend_box("NIFTY 50", results.get('NIFTY 50_trend'))
        with col_inner2:
            simple_trend_box("SENSEX", results.get('SENSEX_trend'))

    with col_center:
        st.write("")  # Empty space in the center
    with col_right:
        st.write("")  # Empty space on right
    result_message(results.get('market_sentiment_text', 'No analysis available'), results.get('score_breakdown', {}).get('market_sentiment_score', 0),10)
    st.markdown("---")
    st.markdown("##### 2. Sector Indices")
    st.caption("Is the sector of the stock currently strong or weak? A strong sector can lift individual stocks within it and vice-versa.")
    result_message(results.get('sector_sentiment_text', 'No analysis available'), results.get('score_breakdown', {}).get('sector_sentiment_score', 0),15)
    st.markdown("---")
    st.markdown("##### 3. Global Cues")
    st.caption("How did the US markets (S&P 500 and NASDAQ) close? Are Asian markets (Nikkei 225 and Hang Seng) showing strength or weakness today?")
    # Create 3 columns and unpack into 3 variables
    col_left, col_center, col_right = st.columns([3, 1, 1])
    with col_left:
        col_inner1, col_inner2,col_inner3,col_inner4 = st.columns(4, gap="small") 
        # Add gap between boxes
        with col_inner1:
            simple_trend_box("S&P 500", results.get('S&P 500_trend'))
        with col_inner2:
            simple_trend_box("NASDAQ", results.get('NASDAQ_trend'))
        with col_inner3:
            simple_trend_box("Nikkei 225", results.get('Nikkei 225_trend')) 
        with col_inner4:
            simple_trend_box("Hang Seng", results.get('Hang Seng_trend'))
    with col_center:
        st.write("")  # Empty space in the center
    with col_right:
        st.write("")  # Empty space on right
    result_message(results.get('global_sentiment_text', 'No analysis available'), results.get('score_breakdown', {}).get('global_sentiment_score', 0),5)
    if show_sentiment and 'company_news_sentiment' in results:       
        st.markdown("---")
        st.subheader("Fundamental & News-Based Analysis")
        st.markdown("##### 1. Company-Specific News")
        result_message(results.get('company_news_sentiment', 'No analysis available'), results.get('score_breakdown', {}).get('company_news_score'),10)
        with st.expander("Show AI News Summary (Groq)"):
                st.markdown(results.get('company_news_summary', 'N/A'))
        st.markdown("---")
            
        # Sector News
        st.markdown("**2. Sector-Specific News**")
        result_message(results.get('sector_news_sentiment', 'No analysis available'), results.get('score_breakdown', {}).get('sector_news_score'),5)
        with st.expander("Show AI Summary & Headlines"):
            st.markdown(results.get('sector_news_summary', 'N/A'))        
        st.markdown("---")
        
        # Macro News
        st.markdown("**3. Macro-Economic News**")
        result_message(results.get('macro_news_sentiment', 'No analysis available'), results.get('score_breakdown', {}).get('macro_news_score'),5)
        with st.expander("Show AI Summary & Headlines"):
            st.markdown(results.get('macro_news_summary', 'N/A'))
    st.markdown("---")
    st.subheader("Final Summary & AI Analysis")
    render_final_score_gauge(results.get('score', 0), results.get('max_score', 100))
    
    with st.expander("View AI Generated Trading Plan", expanded=True):
        st.info(results.get('final_ai_summary', 'Could not generate AI trading plan.'))


def short_term_swing_analysis(ticker):
    """
    Manages the analysis lifecycle, using session_state to avoid re-running slow functions.
    This function acts as the controller.
    """

    # Session State Management: Clear state if the ticker changes
    if 'current_ticker' not in st.session_state or st.session_state.current_ticker != ticker:
        for key in list(st.session_state.keys()):
            if key in ['swing_results', 'news_toggle_state', 'current_ticker']:
                 del st.session_state[key]
        st.session_state.current_ticker = ticker
    
    # Run initial (fast) analysis only if results are not already in memory
    if 'swing_results' not in st.session_state:
        with st.spinner(f"Running initial technical analysis for {ticker}..."):
            analyzer = SwingAnalyzer(ticker)
            st.session_state.swing_results = analyzer.run_initial_analysis()
    
    # Check if initial analysis was successful
    if not st.session_state.swing_results:
        st.error(f"Failed to fetch data or run initial analysis for '{ticker}'. Please ensure the ticker is correct.")
        return
    with st.sidebar:
    # UI Toggle for News Sentiment
        show_sentiment = st.toggle(
            "Analyze News Sentiment (slower)",
            value=st.session_state.get('news_toggle_state', False),
            help="Fetches and analyzes news for the company, its sector, and the macro-economy."
        )
    st.session_state.news_toggle_state = show_sentiment

    # Conditionally run the (slow) news analysis
    # Only runs if the toggle is ON AND we haven't run it before (by checking for a news key)
    if show_sentiment and 'company_news_sentiment' not in st.session_state.swing_results:
        with st.spinner("Fetching and analyzing news sentiment... (this may take a moment)"):
            analyzer = SwingAnalyzer(ticker)
            # Pass the existing results from session state to be updated
            st.session_state.swing_results = analyzer.run_news_analysis(st.session_state.swing_results)

    # Pass the final results (from session_state) to the rendering function
    render_short_term_swing_analysis(st.session_state.swing_results, show_sentiment)



 
def render_prediction_page(ticker="SBIN.NS"):
    """
    Render the prediction page with stock data and AI insights.
    """
    st.title("🤖 AI-Powered Stock Insights")
    with st.sidebar:
         
        st.markdown("## 💼 Investment Type")
        investment_type = st.selectbox(
            "Select Investment Type",
            options=["Long-term","Short-term Swing"],
            index=1
        )  
    if ticker:
        if investment_type == "Short-term Swing":
            short_term_swing_analysis(ticker)
        
        elif investment_type == "Long-term":
            long_term_analysis(ticker)
    else:
        st.warning("Please enter a stock ticker to begin analysis.")