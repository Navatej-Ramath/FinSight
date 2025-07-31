"""
Fibonacci Retracement Analyzer
Advanced technical analysis tool for detecting swing points and calculating Fibonacci levels
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional, Any

class FibonacciAnalyzer:
    """
    Main class for Fibonacci retracement analysis
    """
    
    def __init__(self, min_swing_pct: float = 1.5, scan_window_minutes: int = 90, 
                 high_volume_multiplier: float = 1.5):
        """
        Initialize the Fibonacci analyzer
        
        Args:
            min_swing_pct: Minimum price movement to qualify as a swing
            scan_window_minutes: Scan window for dynamic swing detection
            high_volume_multiplier: Volume multiplier for high volume detection
        """
        self.min_swing_pct = min_swing_pct
        self.scan_window_minutes = scan_window_minutes
        self.high_volume_multiplier = high_volume_multiplier
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove non-trading periods and handle MultiIndex columns
        
        Args:
            df: Raw DataFrame from yfinance
            
        Returns:
            Cleaned DataFrame
        """
        if df is None or df.empty:
            return df
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == df.columns[0][1] else f"{col[0]}_{col[1]}" 
                         for col in df.columns]
        
        # Find volume column
        volume_col = self._find_column(df, 'Volume')
        
        # Filter out non-trading periods
        if volume_col is not None:
            try:
                volume_series = df[volume_col].fillna(0)
                volume_mask = volume_series > 0
                if volume_mask.any():
                    df = df.loc[volume_mask].copy()
            except Exception as e:
                print(f"Volume filtering warning: {e}")
        
        # Remove rows with NaN in OHLC data
        ohlc_cols = []
        for col in df.columns:
            if any(x in str(col).title() for x in ['Open', 'High', 'Low', 'Close']):
                ohlc_cols.append(col)
        
        if ohlc_cols:
            df = df.dropna(subset=ohlc_cols)
        
        return df
    
    def analyze_volume_and_determine_swing_pct(self, df: pd.DataFrame, 
                                             timeframe: str = "15m") -> Tuple[float, bool, Dict]:
        """
        Analyze volume over the scan window and determine appropriate swing percentage
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Chart timeframe (15m, 1h, 1d)
            
        Returns:
            Tuple of (swing_pct, has_high_volume, volume_analysis)
        """
        if df is None or df.empty:
            return self.min_swing_pct, False, {}
        
        volume_col = self._find_column(df, 'Volume')
        if volume_col is None:
            print(f"Warning: Volume column not found. Using default swing percentage: {self.min_swing_pct}%")
            return self.min_swing_pct, False, {}
        
        # Calculate periods in window
        periods_in_window = self._calculate_periods_in_window(timeframe)
        
        # Get the most recent data within the scan window
        recent_data = df.tail(periods_in_window) if len(df) >= periods_in_window else df
        
        # Calculate volume statistics
        avg_volume = df[volume_col].mean()
        recent_avg_volume = recent_data[volume_col].mean()
        max_recent_volume = recent_data[volume_col].max()
        total_recent_volume = recent_data[volume_col].sum()
        
        # Check for high volume activity
        high_volume_threshold = avg_volume * self.high_volume_multiplier
        high_volume_periods = recent_data[recent_data[volume_col] >= high_volume_threshold]
        
        volume_analysis = {
            'avg_volume_overall': avg_volume,
            'recent_avg_volume': recent_avg_volume,
            'max_recent_volume': max_recent_volume,
            'total_recent_volume': total_recent_volume,
            'high_volume_threshold': high_volume_threshold,
            'high_volume_periods': len(high_volume_periods),
            'volume_ratio': recent_avg_volume / avg_volume if avg_volume > 0 else 0
        }
        
        # Determine if there's high volume activity
        has_high_volume = (
            recent_avg_volume >= high_volume_threshold or 
            len(high_volume_periods) >= 2 or
            max_recent_volume >= (avg_volume * 2)
        )
        
        # Set swing percentage based on volume analysis
        if has_high_volume:
            swing_pct = max(1.0, self.min_swing_pct - 0.5)  # Lower threshold for high volume
            print(f"🔥 HIGH VOLUME DETECTED in last {self.scan_window_minutes} minutes!")
            print(f"   Recent avg volume: {recent_avg_volume:,.0f} vs Overall avg: {avg_volume:,.0f}")
            print(f"   High volume periods: {len(high_volume_periods)}")
            print(f"   Setting swing threshold to {swing_pct}% for sensitive detection")
        else:
            swing_pct = self.min_swing_pct
            print(f"📊 Normal volume activity in last {self.scan_window_minutes} minutes")
            print(f"   Recent avg volume: {recent_avg_volume:,.0f} vs Overall avg: {avg_volume:,.0f}")
            print(f"   Using standard swing threshold: {swing_pct}%")
        
        return swing_pct, has_high_volume, volume_analysis
    
    def detect_trend_and_swings_in_window(self, df: pd.DataFrame, timeframe: str = "15m",
                                        min_move_pct: Optional[float] = None) -> Tuple:
        """
        Scan the entire dataset for any window that contains a significant swing
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Chart timeframe
            min_move_pct: Minimum movement percentage (uses default if None)
            
        Returns:
            Tuple of swing analysis results
        """
        if df is None or df.empty:
            return (None,) * 6
        
        if min_move_pct is None:
            min_move_pct = self.min_swing_pct
        
        # Calculate periods in window
        periods_in_window = self._calculate_periods_in_window(timeframe)
        
        if len(df) < periods_in_window:
            return (None,) * 6
        
        # Find column names
        high_col = self._find_column(df, 'High')
        low_col = self._find_column(df, 'Low')
        close_col = self._find_column(df, 'Close')
        volume_col = self._find_column(df, 'Volume')
        
        if not all([high_col, low_col, close_col]):
            print("Error: Could not find required OHLC columns")
            return (None,) * 6
        
        best_swing = None
        best_move_pct = 0
        
        print(f"🔍 Scanning entire dataset for {self.scan_window_minutes}-minute windows with >{min_move_pct}% moves...")
        
        # Scan through all possible windows in the dataset
        for i in range(len(df) - periods_in_window + 1):
            # Extract window
            window_data = df.iloc[i:i + periods_in_window]
            
            if len(window_data) < 2:
                continue
            
            # Find highest high and lowest low in this window
            swing_high = window_data[high_col].max()
            swing_low = window_data[low_col].min()
            
            if pd.isna(swing_high) or pd.isna(swing_low):
                continue
            
            # Calculate move percentage
            move_pct = ((swing_high - swing_low) / swing_low) * 100
            
            if move_pct >= min_move_pct:
                # Get timestamps
                high_time = window_data[window_data[high_col] == swing_high].index[0]
                low_time = window_data[window_data[low_col] == swing_low].index[0]
                
                # Calculate average volume for this window
                window_avg_volume = 0
                if volume_col is not None:
                    window_avg_volume = window_data[volume_col].mean()
                
                # Determine trend direction
                if high_time < low_time:
                    # High came before low = Downtrend
                    trend = "downtrend"
                    fib_start = swing_high
                    fib_end = swing_low
                    start_time = high_time
                    end_time = low_time
                else:
                    # Low came before high = Uptrend
                    trend = "uptrend"
                    fib_start = swing_low
                    fib_end = swing_high
                    start_time = low_time
                    end_time = high_time
                
                # Store this swing if it's the best one found so far
                if move_pct > best_move_pct:
                    best_move_pct = move_pct
                    best_swing = {
                        'fib_start': fib_start,
                        'fib_end': fib_end,
                        'start_time': start_time,
                        'end_time': end_time,
                        'move_pct': move_pct,
                        'trend': trend,
                        'window_start': window_data.index[0],
                        'window_end': window_data.index[-1],
                        'window_avg_volume': window_avg_volume
                    }
                
                print(f"   ✓ Found {trend} swing: {move_pct:.2f}% ({start_time.strftime('%H:%M')} → {end_time.strftime('%H:%M')})")
        
        if best_swing is not None:
            print(f"\n🎯 Best swing found: {best_swing['move_pct']:.2f}% {best_swing['trend']}")
            print(f"   Window: {best_swing['window_start'].strftime('%m-%d %H:%M')} → {best_swing['window_end'].strftime('%m-%d %H:%M')}")
            print(f"   Swing: {best_swing['start_time'].strftime('%H:%M')} → {best_swing['end_time'].strftime('%H:%M')}")
            
            return (best_swing['fib_start'], best_swing['fib_end'], best_swing['start_time'], 
                    best_swing['end_time'], best_swing['move_pct'], best_swing['trend'])
        
        return (None,) * 6
    
    def detect_trend_and_swings_full_dataset(self, df: pd.DataFrame, 
                                           min_move_pct: Optional[float] = None) -> Tuple:
        """
        Identify trend direction and find appropriate swing points (full dataset)
        
        Args:
            df: DataFrame with OHLCV data
            min_move_pct: Minimum movement percentage
            
        Returns:
            Tuple of swing analysis results
        """
        if df is None or df.empty:
            return (None,) * 6
        
        if min_move_pct is None:
            min_move_pct = self.min_swing_pct
        
        # Find column names
        high_col = self._find_column(df, 'High')
        low_col = self._find_column(df, 'Low')
        close_col = self._find_column(df, 'Close')
        
        if not all([high_col, low_col, close_col]):
            print("Error: Could not find required OHLC columns")
            return (None,) * 6
        
        # Find absolute highest high and lowest low
        swing_high = df[high_col].max()
        swing_low = df[low_col].min()
        
        # Check for valid swing
        if pd.isna(swing_high) or pd.isna(swing_low):
            return (None,) * 6
        
        move_pct = ((swing_high - swing_low) / swing_low) * 100
        if move_pct < min_move_pct:
            return (None,) * 6
        
        # Get timestamps
        high_time = df[df[high_col] == swing_high].index[0]
        low_time = df[df[low_col] == swing_low].index[0]
        
        # Determine trend direction based on which came first
        if high_time < low_time:
            # High came before low = Downtrend
            trend = "downtrend"
            fib_start = swing_high
            fib_end = swing_low
            start_time = high_time
            end_time = low_time
        else:
            # Low came before high = Uptrend
            trend = "uptrend"
            fib_start = swing_low
            fib_end = swing_high
            start_time = low_time
            end_time = high_time
        
        return fib_start, fib_end, start_time, end_time, move_pct, trend
    
    def calculate_fibonacci_levels(self, fib_start: float, fib_end: float, 
                                 trend: str) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels based on trend
        
        Args:
            fib_start: Starting price of the swing
            fib_end: Ending price of the swing
            trend: Trend direction ('uptrend' or 'downtrend')
            
        Returns:
            Dictionary of Fibonacci levels
        """
        if trend == "uptrend":
            # In uptrend: retracements go down from high
            fib_levels = {
                '0% (Start - Low)': fib_start,
                '23.6%': fib_start + (fib_end - fib_start) * 0.236,
                '38.2%': fib_start + (fib_end - fib_start) * 0.382,
                '50%': fib_start + (fib_end - fib_start) * 0.5,
                '61.8%': fib_start + (fib_end - fib_start) * 0.618,
                '78.6%': fib_start + (fib_end - fib_start) * 0.786,
                '100% (End - High)': fib_end
            }
        else:
            # In downtrend: retracements go up from low
            fib_levels = {
                '0% (Start - High)': fib_start,
                '23.6%': fib_start - (fib_start - fib_end) * 0.236,
                '38.2%': fib_start - (fib_start - fib_end) * 0.382,
                '50%': fib_start - (fib_start - fib_end) * 0.5,
                '61.8%': fib_start - (fib_start - fib_end) * 0.618,
                '78.6%': fib_start - (fib_start - fib_end) * 0.786,
                '100% (End - Low)': fib_end
            }
        
        return fib_levels
    
    def analyze_current_position(self, current_price: float, fib_levels: Dict[str, float]) -> Dict:
        """
        Analyze current price position relative to Fibonacci levels
        
        Args:
            current_price: Current market price
            fib_levels: Dictionary of Fibonacci levels
            
        Returns:
            Analysis results
        """
        analysis = []
        for level, price in fib_levels.items():
            if price > 0:
                dist_pct = abs(current_price - price) / price * 100
                position = "Above" if current_price > price else "Below"
                analysis.append({
                    'level': level,
                    'price': price,
                    'distance_pct': dist_pct,
                    'position': position
                })
        
        # Find closest level
        closest_level = min(analysis, key=lambda x: x['distance_pct'])
        
        return {
            'levels_analysis': analysis,
            'closest_level': closest_level,
            'current_price': current_price
        }
    
    def _find_column(self, df: pd.DataFrame, column_type: str) -> Optional[str]:
        """
        Find column by type (High, Low, Close, Volume, etc.)
        
        Args:
            df: DataFrame to search
            column_type: Type of column to find
            
        Returns:
            Column name if found, None otherwise
        """
        for col in df.columns:
            if column_type.lower() in str(col).lower():
                return col
        return None
    
    def _calculate_periods_in_window(self, timeframe: str) -> int:
        """
        Calculate number of periods in the scan window based on timeframe
        
        Args:
            timeframe: Chart timeframe (15m, 1h, 1d)
            
        Returns:
            Number of periods
        """
        if timeframe == "15m":
            return self.scan_window_minutes // 15
        elif timeframe == "1h":
            return self.scan_window_minutes // 60
        else:  # 1d
            return max(1, self.scan_window_minutes // (24 * 60))
    
    def get_analysis_summary(self, fib_start: Optional[float], fib_end: Optional[float], 
                           start_time: Optional[datetime], end_time: Optional[datetime],
                           move_pct: Optional[float], trend: Optional[str],
                           current_price: float, volume_analysis: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive analysis summary
        
        Args:
            fib_start: Starting price of swing
            fib_end: Ending price of swing
            start_time: Start time of swing
            end_time: End time of swing
            move_pct: Movement percentage
            trend: Trend direction
            current_price: Current market price
            volume_analysis: Volume analysis results
            
        Returns:
            Analysis summary dictionary
        """
        if fib_start is None:
            return {
                'swing_detected': False,
                'message': 'No significant swing detected',
                'recommendations': [
                    'Wait for a clear breakout or breakdown',
                    'Look for increased volume activity',
                    'Consider other technical analysis tools for ranging markets',
                    'Monitor for news/events that might trigger volatility'
                ]
            }
        
        fib_levels = self.calculate_fibonacci_levels(fib_start, fib_end, trend)
        position_analysis = self.analyze_current_position(current_price, fib_levels)
        
        return {
            'swing_detected': True,
            'trend': trend,
            'move_percentage': move_pct,
            'start_price': fib_start,
            'end_price': fib_end,
            'start_time': start_time,
            'end_time': end_time,
            'fibonacci_levels': fib_levels,
            'position_analysis': position_analysis,
            'volume_analysis': volume_analysis,
            'key_levels': {
                'support': min(fib_levels.values()),
                'resistance': max(fib_levels.values()),
                'fifty_percent': fib_levels['50%']
            }
        }
"""
Fibonacci Retracement Page
Streamlit page for Fibonacci analysis
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from algorithms.fibonacci.fibonacci_analyzer import FibonacciAnalyzer
from algorithms.fibonacci.fibonacci_utils import (
    create_plotly_fibonacci_chart,
    format_levels_for_display,
    get_trading_recommendations
)
from data.data_fetcher import data_fetcher

def render_fibonacci_page(ticker: str = "SBIN.NS"):
    """Render the Fibonacci retracement analysis page"""
    
    st.header("📊 Fibonacci Retracement Analyzer")
    st.markdown("**Advanced swing detection with dynamic volume analysis**")
    
    # Configuration sidebar
    with st.sidebar.expander("⚙️ Analysis Configuration", expanded=True):
        
        
        # Time configuration
        col1, col2 = st.columns(2)
        with col1:
            timeframe_options = data_fetcher.get_timeframe_options()
            timeframe = st.selectbox(
                "⏰ Timeframe",
                options=list(timeframe_options.keys()),
                index=list(timeframe_options.keys()).index("15m"),
                format_func=lambda x: timeframe_options[x]
            )
        
        with col2:
            lookback_days = st.slider(
                "📅 Lookback Days",
                min_value=5,
                max_value=60,
                value=15,
                help="Number of days to analyze"
            )
        
        st.markdown("---")
        
        # Analysis parameters
        st.subheader("🎯 Analysis Parameters")
        
        min_swing_pct = st.slider(
            "Min Swing %",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.1,
            help="Minimum price movement to qualify as a swing"
        )
        
        scan_window = st.slider(
            "Scan Window (min)",
            min_value=30,
            max_value=300,
            value=90,
            step=15,
            help="Window size for dynamic swing detection"
        )
        
        volume_multiplier = st.slider(
            "High Volume Multiplier",
            min_value=1.2,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Volume threshold for high activity detection"
        )
        
        # Analysis type
        analysis_type = st.radio(
            "Analysis Type",
            ["Dynamic Window", "Full Dataset"],
            help="Choose analysis method"
        )
        
        st.markdown("---")
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            show_volume = st.checkbox("Show Volume Analysis", value=True)
            show_recommendations = st.checkbox("Show Trading Recommendations", value=True)
            chart_theme = st.selectbox("Chart Theme", ["plotly_white", "plotly_dark", "simple_white"])
    
    # Analysis button
    analyze_button = st.button(
        "🚀 Run Fibonacci Analysis",
        type="primary",
        use_container_width=True
    )
    
    # Main content area
    if analyze_button:
        run_fibonacci_analysis(
            ticker, timeframe, lookback_days, min_swing_pct,
            scan_window, volume_multiplier, analysis_type,
            show_volume, show_recommendations, chart_theme
        )
    else:
        show_fibonacci_info()

def run_fibonacci_analysis(ticker, timeframe, lookback_days, min_swing_pct,
                          scan_window, volume_multiplier, analysis_type,
                          show_volume, show_recommendations, chart_theme):
    """Run the Fibonacci analysis with given parameters"""
    
    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Fetch data
        status_text.text("📡 Fetching stock data...")
        progress_bar.progress(20)
        
        data, error = data_fetcher.fetch_stock_data(ticker, lookback_days, timeframe)
        
        if error:
            st.error(f"❌ Error fetching data: {error}")
            return
        
        if data is None or data.empty:
            st.error("❌ No data found. Please check the ticker symbol and try again.")
            return
        
        # Step 2: Initialize analyzer
        status_text.text("🔧 Initializing Fibonacci analyzer...")
        progress_bar.progress(40)
        
        analyzer = FibonacciAnalyzer(
            min_swing_pct=min_swing_pct,
            scan_window_minutes=scan_window,
            high_volume_multiplier=volume_multiplier
        )
        
        # Step 3: Clean data
        status_text.text("🧹 Cleaning and processing data...")
        progress_bar.progress(60)
        
        clean_df = analyzer.clean_data(data)
        
        if clean_df.empty:
            st.error("❌ No valid trading data found after cleaning.")
            return
        
        # Step 4: Show basic metrics
        progress_bar.progress(80)
        show_basic_metrics(clean_df, ticker)
        
        # Step 5: Perform analysis
        status_text.text("📊 Performing Fibonacci analysis...")
        progress_bar.progress(90)
        
        if analysis_type == "Dynamic Window":
            perform_dynamic_analysis(analyzer, clean_df, timeframe, ticker, 
                                   show_volume, show_recommendations, chart_theme)
        else:
            perform_full_dataset_analysis(analyzer, clean_df, ticker,
                                        show_volume, show_recommendations, chart_theme)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis completed!")
        
        # Clear progress indicators after a moment
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def show_basic_metrics(df, ticker):
    """Display basic stock metrics"""
    
    # Find close column
    close_col = None
    volume_col = None
    for col in df.columns:
        if 'Close' in str(col).title():
            close_col = col
        elif 'Volume' in str(col).title():
            volume_col = col
    
    if close_col:
        current_price = df[close_col].iloc[-1]
        first_price = df[close_col].iloc[0]
        change = current_price - first_price
        change_pct = (change / first_price) * 100
        
        high_price = df[close_col].max()
        low_price = df[close_col].min()
        avg_volume = df[volume_col].mean() if volume_col else 0
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Current Price", f"₹{current_price:.2f}", f"{change_pct:+.2f}%")
        
        with col2:
            st.metric("Period High", f"₹{high_price:.2f}")
        
        with col3:
            st.metric("Period Low", f"₹{low_price:.2f}")
        
        with col4:
            st.metric("Records", f"{len(df):,}")
        
        with col5:
            if volume_col:
                st.metric("Avg Volume", f"{avg_volume:,.0f}")
            else:
                st.metric("Data Quality", "✅ Good")
        
        # Stock info
        stock_info = data_fetcher.get_stock_info(ticker)
        if stock_info['company_name'] != ticker:
            st.info(f"📊 **{stock_info['company_name']}** | Sector: {stock_info['sector']} | Exchange: {stock_info['exchange']}")

def perform_dynamic_analysis(analyzer, df, timeframe, ticker, show_volume, show_recommendations, chart_theme):
    """Perform dynamic window analysis"""
    
    st.subheader(f"🎯 Dynamic Window Analysis ({analyzer.scan_window_minutes} minutes)")
    
    # Volume analysis
    dynamic_swing_pct, has_high_volume, volume_analysis = analyzer.analyze_volume_and_determine_swing_pct(df, timeframe)
    
    if show_volume and volume_analysis:
        show_volume_analysis(has_high_volume, volume_analysis, dynamic_swing_pct)
    
    # Detect swings
    result = analyzer.detect_trend_and_swings_in_window(df, timeframe, dynamic_swing_pct)
    fib_start, fib_end, start_time, end_time, move_pct, trend = result
    
    if fib_start is not None:
        show_successful_analysis(analyzer, df, fib_start, fib_end, start_time, end_time, 
                                move_pct, trend, ticker, timeframe, volume_analysis, 
                                show_recommendations, chart_theme, is_window_analysis=True)
    else:
        show_failed_analysis(dynamic_swing_pct, analyzer.scan_window_minutes, df)

def perform_full_dataset_analysis(analyzer, df, ticker, show_volume, show_recommendations, chart_theme):
    """Perform full dataset analysis"""
    
    st.subheader("📊 Full Dataset Analysis")
    
    result = analyzer.detect_trend_and_swings_full_dataset(df)
    fib_start, fib_end, start_time, end_time, move_pct, trend = result
    
    if fib_start is not None:
        show_successful_analysis(analyzer, df, fib_start, fib_end, start_time, end_time,
                                move_pct, trend, ticker, "full", None,
                                show_recommendations, chart_theme, is_window_analysis=False)
    else:
        show_failed_analysis(analyzer.min_swing_pct, "full dataset", df)

def show_volume_analysis(has_high_volume, volume_analysis, dynamic_swing_pct):
    """Display volume analysis results"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        volume_status = "🔥 HIGH VOLUME DETECTED" if has_high_volume else "📊 NORMAL VOLUME"
        st.info(f"**Volume Status:** {volume_status}")
        st.write(f"**Dynamic Swing Threshold:** {dynamic_swing_pct}%")
    
    with col2:
        if volume_analysis:
            st.write(f"**Volume Ratio:** {volume_analysis['volume_ratio']:.2f}x")
            st.write(f"**High Volume Periods:** {volume_analysis['high_volume_periods']}")
    
    # Detailed volume metrics
    with st.expander("📈 Detailed Volume Metrics"):
        if volume_analysis:
            vol_col1, vol_col2, vol_col3 = st.columns(3)
            
            with vol_col1:
                st.metric("Overall Avg Volume", f"{volume_analysis['avg_volume_overall']:,.0f}")
            
            with vol_col2:
                st.metric("Recent Avg Volume", f"{volume_analysis['recent_avg_volume']:,.0f}")
            
            with vol_col3:
                st.metric("High Volume Threshold", f"{volume_analysis['high_volume_threshold']:,.0f}")

def show_successful_analysis(analyzer, df, fib_start, fib_end, start_time, end_time,
                           move_pct, trend, ticker, timeframe, volume_analysis,
                           show_recommendations, chart_theme, is_window_analysis):
    """Display successful analysis results"""
    
    st.success(f"✅ **{trend.upper()} SWING DETECTED!**")
    
    # Swing details
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Swing Move", f"{move_pct:.2f}%")
    with col2:
        st.metric("Start Price", f"₹{fib_start:.2f}")
    with col3:
        st.metric("End Price", f"₹{fib_end:.2f}")
    
    # Create and display chart
    fig = create_plotly_fibonacci_chart(
        df, fib_start, fib_end, start_time, end_time,
        move_pct, trend, ticker, timeframe, volume_analysis, is_window_analysis
    )
    fig.update_layout(template=chart_theme)
    st.plotly_chart(fig, use_container_width=True)
    
    # Fibonacci levels analysis
    show_fibonacci_levels_analysis(analyzer, df, fib_start, fib_end, trend)
    
    # Trading recommendations
    if show_recommendations:
        show_trading_recommendations_section(trend, df, analyzer.calculate_fibonacci_levels(fib_start, fib_end, trend))

def show_fibonacci_levels_analysis(analyzer, df, fib_start, fib_end, trend):
    """Display Fibonacci levels analysis"""
    
    st.subheader("📊 Fibonacci Retracement Levels")
    
    # Calculate levels
    fib_levels = analyzer.calculate_fibonacci_levels(fib_start, fib_end, trend)
    
    # Get current price
    close_col = None
    for col in df.columns:
        if 'Close' in str(col).title():
            close_col = col
            break
    
    current_price = df[close_col].iloc[-1] if close_col else 0
    
    # Format and display levels
    levels_df = format_levels_for_display(fib_levels, current_price)
    st.dataframe(levels_df, use_container_width=True, hide_index=True)
    
    # Key insights
    position_analysis = analyzer.analyze_current_position(current_price, fib_levels)
    closest_level = position_analysis['closest_level']
    
    col1, col2 = st.columns(2)
    
    with col1:
        trend_emoji = "📈" if trend == "uptrend" else "📉"
        st.info(f"""
        **{trend_emoji} Trend Analysis:**
        - **Direction:** {trend.title()}
        - **Current Price:** ₹{current_price:.2f}
        - **Support Level:** ₹{min(fib_levels.values()):.2f}
        - **Resistance Level:** ₹{max(fib_levels.values()):.2f}
        """)
    
    with col2:
        st.warning(f"""
        **🎯 Closest Fibonacci Level:**
        - **Level:** {closest_level['level']}
        - **Price:** ₹{closest_level['price']:.2f}
        - **Distance:** {closest_level['distance_pct']:.2f}%
        - **Position:** {closest_level['position']}
        """)

def show_trading_recommendations_section(trend, df, fib_levels):
    """Display trading recommendations"""
    
    st.subheader("💡 Trading Recommendations")
    
    # Get current price
    close_col = None
    for col in df.columns:
        if 'Close' in str(col).title():
            close_col = col
            break
    
    current_price = df[close_col].iloc[-1] if close_col else 0
    
    # Get recommendations
    recommendations = get_trading_recommendations(trend, current_price, fib_levels)
    
    # Display recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **📊 Current Market Position:**
        - **Trend:** {recommendations['trend_direction'].title()}
        - **Position:** {recommendations['current_position'].replace('_', ' ').title()}
        - **Action:** {recommendations['action']}
        """)
    
    with col2:
        st.warning(f"""
        **🎯 Trading Levels:**
        - **Stop Loss:** {recommendations['stop_loss']}
        - **Target:** {recommendations['target']}
        - **50% Level:** ₹{recommendations['key_levels']['fifty_percent']:.2f}
        """)
    
    # General trading tips
    with st.expander("📚 General Trading Tips"):
        for tip in recommendations['general_tips']:
            st.write(f"• {tip}")

def show_failed_analysis(swing_threshold, window_description, df):
    """Display failed analysis message with recommendations"""
    
    st.error("❌ **NO SIGNIFICANT SWING DETECTED**")
    
    st.warning(f"""
    **🚫 FIBONACCI RETRACEMENT NOT RECOMMENDED**
    
    **Analysis Summary:**
    - No significant price swings detected with {swing_threshold}% threshold
    - Market appears to be in consolidation/sideways movement  
    - Low volatility period detected in {window_description}
    
    **Recommendations:**
    - Wait for a clear breakout or breakdown
    - Look for increased volume activity
    - Consider other technical analysis tools for ranging markets
    - Monitor for news/events that might trigger volatility
    """)
    
    # Show current market state
    close_col = None
    for col in df.columns:
        if 'Close' in str(col).title():
            close_col = col
            break
    
    if close_col:
        recent_periods = min(20, len(df))
        recent_data = df.tail(recent_periods)
        recent_high = recent_data[close_col].max()
        recent_low = recent_data[close_col].min()
        range_pct = ((recent_high - recent_low) / recent_low) * 100
        current_price = df[close_col].iloc[-1]
        
        st.info(f"""
        **📈 Current Market State:**
        - **Current Price:** ₹{current_price:.2f}
        - **Recent Range:** ₹{recent_low:.2f} - ₹{recent_high:.2f}
        - **Price Range:** {range_pct:.2f}% (Threshold: {swing_threshold}%)
        - **Market Condition:** Consolidation/Low Volatility
        """)

def show_fibonacci_info():
    """Show information about Fibonacci retracement when not analyzing"""
    
    st.info("👆 Configure your analysis parameters in the sidebar and click **'🚀 Run Fibonacci Analysis'** to begin!")
    
    # Educational content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 About Fibonacci Retracement
        
        Fibonacci retracement is a popular technical analysis tool that uses horizontal lines to indicate areas of support or resistance at key Fibonacci levels before the price continues in the original direction.
        
        **Key Levels:**
        - **23.6%** - Shallow retracement
        - **38.2%** - Moderate retracement  
        - **50.0%** - Psychological level
        - **61.8%** - Golden ratio (most important)
        - **78.6%** - Deep retracement
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 How This Tool Works
        
        **Dynamic Analysis:**
        - Scans recent time windows for significant swings
        - Adjusts sensitivity based on volume activity
        - Identifies the most relevant retracement levels
        
        **Features:**
        - Real-time data from Yahoo Finance
        - Volume-based swing detection
        - Interactive charts with zoom/pan
        - Trading recommendations
        - Multiple timeframe support
        """)
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    
    steps_col1, steps_col2 = st.columns(2)
    
    with steps_col1:
        st.markdown("""
        **Step 1: Select Stock**
        - Enter NSE ticker (e.g., SBIN.NS)
        - Or choose from popular stocks dropdown
        
        **Step 2: Configure Timeframe**
        - Choose timeframe (15m for intraday, 1d for swing)
        - Set lookback period (15 days recommended)
        """)
    
    with steps_col2:
        st.markdown("""
        **Step 3: Adjust Parameters**
        - Set minimum swing percentage (1.5% default)
        - Configure scan window (90 minutes default)
        - Choose analysis type (Dynamic/Full Dataset)
        
        **Step 4: Run Analysis**
        - Click 'Run Fibonacci Analysis'
        - Review results and trading levels
        """)

# Export the render function
__all__ = ['render_fibonacci_page']