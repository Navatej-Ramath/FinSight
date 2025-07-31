"""
Stock Visualization Page
Comprehensive stock analysis and visualization tool
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.data_fetcher import data_fetcher
from utils.config import DATA_CONFIG, UI_CONFIG, APP_CONFIG


def render_stock_visualization_page(ticker: str = "SBIN.NS"):
    """Render the stock visualization and analysis page"""
    
    
    # Configuration sidebar
    with st.sidebar.expander("⚙️ Visualization Settings", expanded=True):
        
        # Stock selection
        ticker_input = ticker
        
        # Time period selection
        period_options = {
            "1mo": "1 Month",
            "3mo": "3 Months", 
            "6mo": "6 Months",
            "1y": "1 Year",
            "2y": "2 Years",
            "5y": "5 Years",
            "max": "Maximum Available"
        }
        
        selected_period = st.selectbox(
            "📅 Time Period",
            options=list(period_options.keys()),
            index=3,  # Default to 1 year
            format_func=lambda x: period_options[x]
        )
        
        # Chart options
        st.subheader("📊 Chart Options")
        
        show_volume = st.checkbox("Show Volume", value=False)
        show_bollinger_bands = st.checkbox("Show Bollinger Bands", value=False)
    run_stock_analysis(
        ticker_input, selected_period, show_volume,
        show_bollinger_bands
    )

def run_stock_analysis(ticker, period, show_volume,
                      show_bollinger_bands):
    """Run comprehensive stock analysis"""
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Fetch stock data
        status_text.text("📡 Fetching stock data...")
        progress_bar.progress(20)
        
        stock_data, company_info = data_fetcher.fetch_stock_data_and_info(ticker, period)
        
        if stock_data is None:
            st.error("❌ Could not fetch stock data. Please check the ticker symbol.")
            return
        
        progress_bar.progress(40)
        
        # Step 2: Display company information
        if company_info:
            status_text.text("🏢 Displaying company information...")
            display_company_info(company_info, ticker)
            progress_bar.progress(60)
        
        # Step 3: Display key metrics
        
        status_text.text("📊 Calculating key metrics...")
        display_key_metrics(stock_data, company_info)
        progress_bar.progress(70)
        
        # Step 4: Generate charts
        status_text.text("📈 Generating interactive charts...")
        
        # Main price chart
        create_main_price_chart(stock_data, ticker, show_volume, 
                                show_bollinger_bands)
        
        progress_bar.progress(85)
        
        # Additional analysis charts
        create_additional_charts(stock_data, ticker)
        
        progress_bar.progress(95)
        
        # Step 5: Price prediction (if enabled)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis completed!")
        
        # Clear progress indicators
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        progress_bar.empty()
        status_text.empty()



def display_company_info(company_info, ticker):
    """Display company information"""
    
    st.subheader(f"🏢 {company_info.get('shortName', ticker)} Company Information")
    
    # Company overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if 'longBusinessSummary' in company_info:
            st.write("**Business Summary:**")
            st.write(company_info['longBusinessSummary'][:500] + "..." if len(company_info['longBusinessSummary']) > 500 else company_info['longBusinessSummary'])
        
        if 'website' in company_info:
            st.write(f"**Website:** [{company_info['website']}]({company_info['website']})")
    
    with col2:
        # Company details
        details = {}
        
        if 'sector' in company_info:
            details['Sector'] = company_info['sector']
        if 'industry' in company_info:
            details['Industry'] = company_info['industry']
        if 'country' in company_info:
            details['Country'] = company_info['country']
        if 'fullTimeEmployees' in company_info:
            details['Employees'] = f"{company_info['fullTimeEmployees']:,}"
        if 'marketCap' in company_info:
            details['Market Cap'] = f"₹{company_info['marketCap']:,.0f}" if 'NS' in ticker else f"${company_info['marketCap']:,.0f}"
        
        if details:
            details_df = pd.DataFrame(list(details.items()), columns=['Metric', 'Value'])
            st.table(details_df.set_index('Metric'))


def display_key_metrics(stock_data, company_info):
    """Display key financial metrics similar to Sona BLW format"""
    
    st.subheader("📊 Key Metrics")
    
    # Calculate metrics from stock data
    current_price = stock_data['Close'].iloc[-1]
    previous_price = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
    price_change = current_price - previous_price
    price_change_pct = (price_change / previous_price) * 100
    
    max_price = stock_data['High'].max()
    min_price = stock_data['Low'].min()
    avg_volume = stock_data['Volume'].mean()
    # First row - Basic price metrics (similar to left panel in image)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"₹{current_price:.2f}", f"{price_change_pct:+.2f}%")
    
    with col2:
        high_low_range = f"₹{max_price:.0f} / {min_price:.0f}"
        st.metric("High / Low", high_low_range)
    
    with col3:
        st.metric("Day High", f"₹{stock_data['High'].iloc[-1]:.2f}")
    
    with col4:
        st.metric("Day Low", f"₹{stock_data['Low'].iloc[-1]:.2f}")
    
    # Second row - Financial ratios (similar to right panel metrics)
    if company_info:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Market Cap (similar to image showing ₹28,944 Cr)
            if 'marketCap' in company_info:
                market_cap = company_info['marketCap'] / 1e7  # Convert to Crores
                st.metric("Market Cap", f"₹{market_cap:,.0f} Cr.")
            else:
                st.metric("Volume", f"{avg_volume:,.0f}")
        
        with col2:
            # Book Value (similar to image showing ₹88.4)
            if 'bookValue' in company_info:
                st.metric("Book Value", f"₹{company_info['bookValue']:.1f}")
            elif 'priceToBook' in company_info and company_info['priceToBook']:
                book_value = current_price / company_info['priceToBook']
                st.metric("Book Value", f"₹{book_value:.1f}")
            else:
                st.metric("52W High", f"₹{max_price:.2f}")
        
        with col3:
            # P/E Ratio (similar to image showing 49.6)
            if 'trailingPE' in company_info and company_info['trailingPE']:
                st.metric("Stock P/E", f"{company_info['trailingPE']:.1f}")
            elif 'forwardPE' in company_info and company_info['forwardPE']:
                st.metric("Forward P/E", f"{company_info['forwardPE']:.1f}")
            else:
                st.metric("52W Low", f"₹{min_price:.2f}")
        
        with col4:
            # Dividend Yield (similar to image showing 0.69%)
            if 'dividendYield' in company_info and company_info['dividendYield']:
                dividend_yield = company_info['dividendYield'] * 100
                st.metric("Dividend Yield", f"{dividend_yield:.2f}%")
            else:
                st.metric("Avg Volume", f"{avg_volume:,.0f}")
    

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # ROCE (similar to image showing 17.8%)
        if company_info and 'returnOnEquity' in company_info and company_info['returnOnEquity']:
            roe = company_info['returnOnEquity'] * 100
            st.metric("ROE", f"{roe:.1f}%")
        else:
            st.metric("Price Change", f"₹{price_change:+.2f}")
    
    with col2:
        # ROE (similar to image showing 14.3%)
        if company_info and 'returnOnAssets' in company_info and company_info['returnOnAssets']:
            roa = company_info['returnOnAssets'] * 100
            st.metric("ROA", f"{roa:.1f}%")
        else:
            volatility = stock_data['Close'].pct_change().std() * 100
            st.metric("Volatility", f"{volatility:.1f}%")
    
    with col3:
        # Show Face Value if available, else show today's Volume
        if 'faceValue' in company_info:
            st.metric("Face Value", f"₹{company_info['faceValue']:.1f}")
        else:
            st.metric("Today's Volume", f"{stock_data['Volume'].iloc[-1]:,.0f}")

    with col4:   
        if company_info and 'trailingEps' in company_info:
            st.metric("EPS (TTM)", f"₹{company_info['trailingEps']:.2f}")
        else:
            st.metric("EPS (TTM)", "Data Unavailable")

    
def display_company_overview(company_info):
    """Display company overview section similar to the 'ABOUT' section"""
    
    if not company_info:
        return
    
    st.markdown("### About the Company")
    
    # Company description
    if 'longBusinessSummary' in company_info:
        st.write(company_info['longBusinessSummary'])
    
    # Key company details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Key Details:**")
        if 'industry' in company_info:
            st.write(f"**Industry:** {company_info['industry']}")
        if 'sector' in company_info:
            st.write(f"**Sector:** {company_info['sector']}")
        if 'country' in company_info:
            st.write(f"**Country:** {company_info['country']}")
    
    with col2:
        st.markdown("**Financial Highlights:**")
        if 'totalRevenue' in company_info and company_info['totalRevenue']:
            revenue_cr = company_info['totalRevenue'] / 1e7  # Convert to Crores
            st.write(f"**Revenue:** ₹{revenue_cr:,.0f} Cr.")
        if 'totalCash' in company_info and company_info['totalCash']:
            cash_cr = company_info['totalCash'] / 1e7
            st.write(f"**Cash:** ₹{cash_cr:,.0f} Cr.")
        if 'totalDebt' in company_info and company_info['totalDebt']:
            debt_cr = company_info['totalDebt'] / 1e7
            st.write(f"**Total Debt:** ₹{debt_cr:,.0f} Cr.")
def create_main_price_chart(stock_data, ticker, show_volume, show_bollinger_bands):
    """Create main price chart with indicators"""
    
    # Create subplots
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price Chart', 'Volume'),
            row_heights=[0.7, 0.3]
        )
    else:
        fig = make_subplots(rows=1, cols=1)
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=stock_data['Date'],
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name='Price',
            showlegend=False
        ),
        row=1, col=1
    )

    # Bollinger Bands
    if show_bollinger_bands:
        period = 20
        if len(stock_data) >= period:
            rolling_mean = stock_data['Close'].rolling(window=period).mean()
            rolling_std = stock_data['Close'].rolling(window=period).std()
            upper_band = rolling_mean + (rolling_std * 2)
            lower_band = rolling_mean - (rolling_std * 2)
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data['Date'],
                    y=upper_band,
                    mode='lines',
                    name='Upper BB',
                    line=dict(color='gray', dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data['Date'],
                    y=lower_band,
                    mode='lines',
                    name='Lower BB',
                    line=dict(color='gray', dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)',
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # Volume chart
    if show_volume:
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(stock_data['Close'], stock_data['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=stock_data['Date'],
                y=stock_data['Volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False,
                opacity=0.7
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} - Stock Price Analysis',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)


def create_additional_charts(stock_data, ticker):
    """Create additional analysis charts"""
    
    st.subheader("📈 Additional Analysis")
    
    # Create tabs for different charts
    tab1, tab2, tab3 = st.tabs(["📊 Price vs Moving Averages", "📈 Daily Returns", "📉 Price Distribution"])
    
    with tab1:
        # Price with multiple moving averages
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=stock_data['Date'],
            y=stock_data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add moving averages
        for period, color in [(20, 'red'), (50, 'green'), (100, 'orange'), (200, 'purple')]:
            if len(stock_data) >= period:
                ma = stock_data['Close'].rolling(window=period).mean()
                fig.add_trace(go.Scatter(
                    x=stock_data['Date'],
                    y=ma,
                    mode='lines',
                    name=f'MA {period}',
                    line=dict(color=color, width=1)
                ))
        
        fig.update_layout(
            title=f'{ticker} - Price with Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Daily returns
        stock_data['Daily_Return'] = stock_data['Close'].pct_change() * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stock_data['Date'],
            y=stock_data['Daily_Return'],
            mode='lines',
            name='Daily Returns (%)',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title=f'{ticker} - Daily Returns',
            xaxis_title='Date',
            yaxis_title='Daily Return (%)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Returns statistics
        returns_stats = {
            'Mean Return': f"{stock_data['Daily_Return'].mean():.3f}%",
            'Std Deviation': f"{stock_data['Daily_Return'].std():.3f}%",
            'Max Return': f"{stock_data['Daily_Return'].max():.3f}%",
            'Min Return': f"{stock_data['Daily_Return'].min():.3f}%"
        }
        
        stats_df = pd.DataFrame(list(returns_stats.items()), columns=['Metric', 'Value'])
        st.table(stats_df.set_index('Metric'))
    
    with tab3:
        # Price distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=stock_data['Close'],
            nbinsx=50,
            name='Price Distribution',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title=f'{ticker} - Price Distribution',
            xaxis_title='Price (₹)',
            yaxis_title='Frequency',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)





# Export the render function
__all__ = ['render_stock_visualization_page']