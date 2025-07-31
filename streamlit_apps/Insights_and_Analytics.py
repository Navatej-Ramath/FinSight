"""
Main Streamlit Application
Entry point for the Technical Analysis Suite
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from pages.stock_visualization_page import render_stock_visualization_page
from algorithms.fibonacci.fibonacci_analyzer import render_fibonacci_page
from data.data_fetcher import data_fetcher
from pages.Prediction import render_prediction_page

# Page configuration
st.set_page_config(
    page_title="FinSight Trading Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

def insights_analytics_page():
    """Insights and Analytics page function"""
    
    # Get ticker from session state or use default
    ticker = st.session_state.get('selected_ticker', 'SBIN.NS')
    
    # Algorithm selection for this page
    st.sidebar.subheader("🔧 Select Analysis Algorithm")

    algorithms = {
        "Fibonacci Retracement": {
            "icon": "📊",
            "description": "Advanced Fibonacci analysis with dynamic swing detection",
            "status": "✅ Available"
        },
        "RSI Analysis": {
            "icon": "📈",
            "description": "Relative Strength Index with divergence detection",
            "status": "🚧 Coming Soon"
        },
        "Bollinger Bands": {
            "icon": "📉",
            "description": "Volatility analysis with squeeze detection",
            "status": "🚧 Coming Soon"
        },
        "MACD Analysis": {
            "icon": "🌊",
            "description": "Moving Average Convergence Divergence signals",
            "status": "🚧 Coming Soon"
        },
        "Support & Resistance": {
            "icon": "🏗️",
            "description": "Automatic level detection with volume confirmation",
            "status": "🚧 Coming Soon"
        }
    }

    selected_algorithm = st.sidebar.selectbox(
        "Choose Algorithm:",
        options=list(algorithms.keys()),
        format_func=lambda x: f"{algorithms[x]['icon']} {x}"
    )

    
    
    # Render the selected algorithm
    if selected_algorithm == "Fibonacci Retracement":
        render_fibonacci_page(ticker)
    else:
        st.warning(f"🚧 {selected_algorithm} is coming soon!")

def prediction_page_wrapper():
    # Get ticker from session state
    ticker = st.session_state.get('selected_ticker', 'SBIN.NS')
    render_prediction_page(ticker)
    
 

def stock_visualization_page_wrapper():
    """Stock Visualization page wrapper"""
    ticker = st.session_state.get('selected_ticker', 'SBIN.NS')
    render_stock_visualization_page(ticker)

def main():
    """Main application function"""

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .market-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .market-open {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .market-closed {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>FinSight Trader</h1>
        <p>Let the stocks burn 🔥</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Define pages for navigation
    pages = [
        st.Page(insights_analytics_page, title="Insights and Analytics", icon="📊"),
        st.Page(prediction_page_wrapper, title="Prediction", icon="🔮"),
        st.Page(stock_visualization_page_wrapper, title="Stock Visualization", icon="📈")
    ]
    
    # Create navigation
    pg = st.navigation(pages)
    
    # Sidebar content (will be available on all pages)
    with st.sidebar:
        st.title("⚙️ Control Panel")
        
        # Market status
        market_status = data_fetcher.check_market_status()
        status_class = "market-open" if market_status['is_open'] else "market-closed"
        
        st.markdown(f"""
        <div class="market-status {status_class}">
            🕐 Market Status: {market_status['status']}<br>
            <small>{market_status['next_open']}</small>
        </div>
        """, unsafe_allow_html=True)
        
        
        # Stock selection
        st.markdown("### 📈 Stock Selection")
        ticker = st.text_input(
            "Stock Ticker (NSE)",
            value=st.session_state.get('selected_ticker', 'SBIN.NS'),
            help="Enter NSE ticker symbol (e.g., SBIN.NS, RELIANCE.NS)",
            key="ticker_input"
        )
        
        # Quick stock selection
        popular_stocks = data_fetcher.get_popular_nse_stocks()
        selected_popular = st.selectbox(
            "Or choose from popular stocks:",
            options=["Custom"] + list(popular_stocks.keys()),
            format_func=lambda x: f"{x} - {popular_stocks.get(x, '')}" if x != "Custom" else "Custom Ticker"
        )
        
        # Override ticker if a popular one is selected
        if selected_popular != "Custom":
            ticker = selected_popular
        
        # Store ticker in session state for all pages to access
        st.session_state['selected_ticker'] = ticker
        
        st.markdown("---")
        
        # Current selection info
        st.markdown("### 📋 Current Selection")
        st.write(f"**Stock:** {ticker}")
        
        
    # Run the selected page
    pg.run()
    
    # Footer (will appear on all pages)
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>🚀 <strong>Technical Analysis Suite</strong> | Built with Streamlit & Python</p>
        <p><em>For educational and analysis purposes only. Not financial advice.</em></p>
        <p><small>Real-time data provided by Yahoo Finance API</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()