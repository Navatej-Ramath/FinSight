import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

# Configuration
TICKER = "SBIN.NS"  # SBI stock on NSE
INTRADAY_TIMEFRAME = "15m"  # 15-minute candles for intraday
SWING_ANALYSIS_DAYS = 15  # Look back 15 days for swing points
MIN_SWING_MOVE_PCT = 1.5  # Default minimum price movement to qualify as a swing
SCAN_WINDOW_MINUTES = 90  # Scan window for dynamic swing detection
HIGH_VOLUME_MULTIPLIER = 1.5  # Volume must be X times average to be considered high

def fetch_stock_data():
    """Fetch historical data from Yahoo Finance with multiple timeframes"""
    end_date = datetime.now()
    
    # If it's weekend, use last Friday as end date
    if end_date.weekday() >= 5:  # Saturday=5, Sunday=6
        days_back = end_date.weekday() - 4  # Go back to Friday
        end_date = end_date - timedelta(days=days_back)
    
    start_date = end_date - timedelta(days=SWING_ANALYSIS_DAYS)
    
    # Fetch data with auto_adjust explicitly set to avoid warnings
    intraday_data = yf.download(
        TICKER,
        start=start_date,
        end=end_date,
        interval=INTRADAY_TIMEFRAME,
        progress=False,
        auto_adjust=True
    )
    
    daily_data = yf.download(
        TICKER,
        start=start_date - timedelta(days=30),  # Extra buffer
        end=end_date,
        interval="1d",
        progress=False,
        auto_adjust=True
    )
    
    return intraday_data, daily_data

def clean_data(df):
    """Remove non-trading periods and handle MultiIndex columns"""
    if df.empty:
        return df
    
    # Handle MultiIndex columns (common with yfinance data)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == df.columns[0][1] else f"{col[0]}_{col[1]}" for col in df.columns]
    
    # Find volume column
    volume_col = None
    for col in df.columns:
        if 'Volume' in str(col).title():
            volume_col = col
            break
    
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

def analyze_volume_and_determine_swing_pct(df, window_minutes=SCAN_WINDOW_MINUTES):
    """
    Analyze volume over the scan window and determine appropriate swing percentage
    """
    if df.empty:
        return MIN_SWING_MOVE_PCT, False, {}
    
    # Find volume column
    volume_col = None
    for col in df.columns:
        if 'Volume' in str(col).title():
            volume_col = col
            break
    
    if volume_col is None:
        print(f"Warning: Volume column not found. Using default swing percentage: {MIN_SWING_MOVE_PCT}%")
        return MIN_SWING_MOVE_PCT, False, {}
    
    # Calculate how many 15-minute periods are in the scan window
    periods_in_window = window_minutes // 15
    
    # Get the most recent data within the scan window
    recent_data = df.tail(periods_in_window) if len(df) >= periods_in_window else df
    
    # Calculate volume statistics
    avg_volume = df[volume_col].mean()
    recent_avg_volume = recent_data[volume_col].mean()
    max_recent_volume = recent_data[volume_col].max()
    total_recent_volume = recent_data[volume_col].sum()
    
    # Check for high volume activity
    high_volume_threshold = avg_volume * HIGH_VOLUME_MULTIPLIER
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
        swing_pct = 1.0  # Lower threshold for high volume periods
        print(f"🔥 HIGH VOLUME DETECTED in last {window_minutes} minutes!")
        print(f"   Recent avg volume: {recent_avg_volume:,.0f} vs Overall avg: {avg_volume:,.0f}")
        print(f"   High volume periods: {len(high_volume_periods)}")
        print(f"   Setting swing threshold to {swing_pct}% for sensitive detection")
    else:
        swing_pct = MIN_SWING_MOVE_PCT
        print(f"📊 Normal volume activity in last {window_minutes} minutes")
        print(f"   Recent avg volume: {recent_avg_volume:,.0f} vs Overall avg: {avg_volume:,.0f}")
        print(f"   Using standard swing threshold: {swing_pct}%")
    
    return swing_pct, has_high_volume, volume_analysis

def detect_trend_and_swings_in_window(df, window_minutes=SCAN_WINDOW_MINUTES, min_move_pct=MIN_SWING_MOVE_PCT):
    """
    Scan the entire dataset for any 90-minute window that contains a significant swing
    """
    if df.empty:
        return None, None, None, None, None, None
    
    # Calculate how many 15-minute periods are in the scan window
    periods_in_window = window_minutes // 15
    
    if len(df) < periods_in_window:
        return None, None, None, None, None, None
    
    # Find column names
    high_col = low_col = close_col = volume_col = None
    for col in df.columns:
        col_str = str(col).title()
        if 'High' in col_str:
            high_col = col
        elif 'Low' in col_str:
            low_col = col
        elif 'Close' in col_str:
            close_col = col
        elif 'Volume' in col_str:
            volume_col = col
    
    if not all([high_col, low_col, close_col]):
        print("Error: Could not find required OHLC columns")
        return None, None, None, None, None, None
    
    best_swing = None
    best_move_pct = 0
    
    print(f"🔍 Scanning entire dataset for {window_minutes}-minute windows with >{min_move_pct}% moves...")
    
    # Scan through all possible 90-minute windows in the dataset
    for i in range(len(df) - periods_in_window + 1):
        # Extract 90-minute window
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
    
    return None, None, None, None, None, None

def detect_trend_and_swings(df, min_move_pct=MIN_SWING_MOVE_PCT):
    """Identify trend direction and find appropriate swing points (full dataset)"""
    if df.empty:
        return None, None, None, None, None, None
    
    # Find column names
    high_col = low_col = close_col = None
    for col in df.columns:
        col_str = str(col).title()
        if 'High' in col_str:
            high_col = col
        elif 'Low' in col_str:
            low_col = col
        elif 'Close' in col_str:
            close_col = col
    
    if not all([high_col, low_col, close_col]):
        print("Error: Could not find required OHLC columns")
        return None, None, None, None, None, None
    
    # Find absolute highest high and lowest low
    swing_high = df[high_col].max()
    swing_low = df[low_col].min()
    
    # Check for valid swing
    if pd.isna(swing_high) or pd.isna(swing_low):
        return None, None, None, None, None, None
    
    move_pct = ((swing_high - swing_low) / swing_low) * 100
    if move_pct < min_move_pct:
        return None, None, None, None, None, None
    
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

def calculate_fibonacci_levels(fib_start, fib_end, trend):
    """Calculate Fibonacci retracement levels based on trend"""
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

def plot_fibonacci_clean(df, fib_start, fib_end, start_time, end_time, move_pct, trend, volume_analysis=None, is_window_analysis=False):
    """Plot Fibonacci retracement with clean time axis (no gaps)"""
    # Find close column
    close_col = None
    for col in df.columns:
        if 'Close' in str(col).title():
            close_col = col
        if 'High' in str(col).title():
            high_col = col
        if 'Low' in str(col).title():
            low_col = col
    
    if close_col is None:
        print("Error: Could not find Close column")
        return
    
    # Calculate Fibonacci levels
    fib_levels = calculate_fibonacci_levels(fib_start, fib_end, trend)
    
    # Create clean time series for plotting (no gaps)
    df_clean = df.copy()
    df_clean['plot_x'] = range(len(df_clean))  # Sequential numbering
    
    # Find plot positions for swing points
    start_plot_x = df_clean[df_clean.index == start_time]['plot_x'].iloc[0]
    end_plot_x = df_clean[df_clean.index == end_time]['plot_x'].iloc[0]
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[3, 1])
    
    # Price chart
    ax1.plot(df_clean['plot_x'], df_clean[close_col], 
             label='Price', color='navy', linewidth=1.5)
    ax1.plot(df_clean['plot_x'], df_clean[low_col], 
             label='Price', color='red', linewidth=1.5)
    ax1.plot(df_clean['plot_x'], df_clean[high_col], 
             label='Price', color='green', linewidth=1.5)
    # Swing markers
    ax1.scatter(start_plot_x, fib_start, color='blue', s=150, 
               label=f'Swing Start ({trend.title()})', zorder=5)
    ax1.scatter(end_plot_x, fib_end, color='red', s=150, 
               label=f'Swing End ({trend.title()})', zorder=5)
    
    # Fibonacci trend line
    ax1.plot([start_plot_x, end_plot_x], [fib_start, fib_end], 
             color='black', linewidth=2, linestyle='-', alpha=0.8, 
             label='Fibonacci Trend Line')
    
    # Fibonacci retracement levels
    colors = ['red', 'orange', 'gold', 'green', 'teal', 'purple', 'blue']
    for (level, price), color in zip(fib_levels.items(), colors):
        ax1.axhline(price, color=color, linestyle='--', alpha=0.7, 
                   label=f'Fib {level}', linewidth=1.5)
    
    # Volume chart
    volume_col = None
    for col in df.columns:
        if 'Volume' in str(col).title():
            volume_col = col
            break
    
    if volume_col is not None:
        bars = ax2.bar(df_clean['plot_x'], df_clean[volume_col], 
                      color='lightblue', alpha=0.7, label='Volume')
        
        # Highlight high volume periods if analysis is available
        if volume_analysis and 'high_volume_threshold' in volume_analysis:
            high_vol_threshold = volume_analysis['high_volume_threshold']
            high_vol_mask = df_clean[volume_col] >= high_vol_threshold
            if high_vol_mask.any():
                ax2.bar(df_clean[high_vol_mask]['plot_x'], df_clean[high_vol_mask][volume_col], 
                       color='red', alpha=0.8, label='High Volume')
                ax2.axhline(high_vol_threshold, color='red', linestyle=':', 
                           alpha=0.7, label=f'High Vol Threshold')
        
        ax2.set_ylabel('Volume', fontsize=10)
        ax2.legend(loc='upper right')
    
    # Custom x-axis labels
    num_labels = min(10, len(df_clean))
    step = max(1, len(df_clean) // num_labels)
    tick_positions = list(range(0, len(df_clean), step))
    tick_labels = [df_clean.index[i].strftime('%m-%d %H:%M') for i in tick_positions]
    
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45)
    
    # Formatting
    trend_arrow = "↗" if trend == "uptrend" else "↘"
    window_text = f" ({SCAN_WINDOW_MINUTES}min window)" if is_window_analysis else ""
    title = f'SBI Fibonacci Retracement | {trend.title()} {trend_arrow} | {INTRADAY_TIMEFRAME} Data | {move_pct:.2f}% Move{window_text}'
    
    ax1.set_title(title, pad=20, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (₹)', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax1.grid(alpha=0.3)
    
    ax2.set_xlabel('Time (No Gaps)', fontsize=12)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print levels with current price analysis
    current_price = df_clean[close_col].iloc[-1]
    analysis_type = "WINDOW" if is_window_analysis else "FULL DATASET"
    print(f"\n{trend.upper()} Fibonacci Retracement Analysis ({analysis_type}):")
    print(f"Trend Direction: {trend.title()} {trend_arrow}")
    print(f"Current Price: ₹{current_price:.2f}")
    print(f"Swing Move: {move_pct:.2f}% ({fib_start:.2f} → {fib_end:.2f})")
    
    if volume_analysis:
        print(f"\nVolume Analysis:")
        print(f"Overall Avg Volume: {volume_analysis['avg_volume_overall']:,.0f}")
        print(f"Recent Avg Volume: {volume_analysis['recent_avg_volume']:,.0f}")
        print(f"Volume Ratio: {volume_analysis['volume_ratio']:.2f}x")
        print(f"High Volume Periods: {volume_analysis['high_volume_periods']}")
    
    print(f"\nFibonacci Levels:")
    for level, price in fib_levels.items():
        dist_pct = abs(current_price - price)/price * 100
        position = "Above" if current_price > price else "Below"
        print(f"{level:<18}: ₹{price:.2f} | {position} by {dist_pct:.2f}%")

# Main execution
try:
    # Check if it's weekend
    today = datetime.now()
    if today.weekday() >= 5:
        print(f"Note: Today is {'Saturday' if today.weekday() == 5 else 'Sunday'}. Indian markets are closed.")
        print("Using data up to last Friday for analysis.\n")
    
    print(f"Fetching {SWING_ANALYSIS_DAYS} days of {INTRADAY_TIMEFRAME} data for {TICKER}...")
    intraday_data, daily_data = fetch_stock_data()
    
    # Clean the data
    intraday_clean = clean_data(intraday_data)
    daily_clean = clean_data(daily_data)
    
    # Check if data was fetched successfully
    if intraday_clean.empty and daily_clean.empty:
        print("No valid trading data found. Please check your internet connection and ticker symbol.")
    else:
        print(f"Processed {len(intraday_clean)} intraday records and {len(daily_clean)} daily records")
        
        if not intraday_clean.empty:
            print(f"\n{'='*60}")
            print(f"PHASE 1: DYNAMIC SWING DETECTION ({SCAN_WINDOW_MINUTES}-MINUTE WINDOW)")
            print(f"{'='*60}")
            
            # Analyze volume and determine appropriate swing percentage
            dynamic_swing_pct, has_high_volume, volume_analysis = analyze_volume_and_determine_swing_pct(intraday_clean)
            
            # Try window-based analysis first
            fib_start, fib_end, start_time, end_time, move_pct, trend = detect_trend_and_swings_in_window(
                intraday_clean, SCAN_WINDOW_MINUTES, dynamic_swing_pct
            )
            
            if fib_start is not None:
                print(f"\n✅ SWING DETECTED in {SCAN_WINDOW_MINUTES}-minute window!")
                print(f"   {trend.title()} move: {move_pct:.2f}%")
                print(f"   Start: ₹{fib_start:.2f} at {start_time}")
                print(f"   End:   ₹{fib_end:.2f} at {end_time}")
                plot_fibonacci_clean(intraday_clean, fib_start, fib_end, start_time, end_time, 
                                   move_pct, trend, volume_analysis, is_window_analysis=True)
            else:
                print(f"\n❌ No swing found in {SCAN_WINDOW_MINUTES}-minute window with {dynamic_swing_pct}% threshold")
                print(f"\n{'⚠️ '*20}")
                print(f"🚫 FIBONACCI RETRACEMENT TOOL NOT RECOMMENDED")
                print(f"{'⚠️ '*20}")
                print(f"\n📊 ANALYSIS SUMMARY:")
                print(f"   • No significant price swings detected in the last {SCAN_WINDOW_MINUTES} minutes")
                print(f"   • Current market conditions appear to be:")
                print(f"     - Consolidating/sideways movement")
                print(f"     - Low volatility period")
                print(f"     - Insufficient momentum for reliable Fibonacci analysis")
                print(f"\n💡 RECOMMENDATIONS:")
                print(f"   • Wait for a clear breakout or breakdown")
                print(f"   • Look for increased volume activity")
                print(f"   • Consider other technical analysis tools for ranging markets")
                print(f"   • Monitor for news/events that might trigger volatility")
                print(f"\n🔄 RETRY CONDITIONS:")
                print(f"   • Volume spike (>1.5x average)")
                print(f"   • Price movement >{dynamic_swing_pct}% in {SCAN_WINDOW_MINUTES} minutes")
                print(f"   • Clear trend formation")
                
                # Show current market state
                close_col = None
                for col in intraday_clean.columns:
                    if 'Close' in str(col).title():
                        close_col = col
                        break
                
                if close_col is not None:
                    current_price = intraday_clean[close_col].iloc[-1]
                    periods_in_window = SCAN_WINDOW_MINUTES // 15
                    window_data = intraday_clean.tail(periods_in_window) if len(intraday_clean) >= periods_in_window else intraday_clean
                    window_high = window_data[close_col].max()
                    window_low = window_data[close_col].min()
                    window_range_pct = ((window_high - window_low) / window_low) * 100
                    
                    print(f"\n📈 CURRENT MARKET STATE:")
                    print(f"   • Current Price: ₹{current_price:.2f}")
                    print(f"   • {SCAN_WINDOW_MINUTES}min Range: ₹{window_low:.2f} - ₹{window_high:.2f}")
                    print(f"   • Price Range: {window_range_pct:.2f}% (Threshold: {dynamic_swing_pct}%)")
                    
                    if volume_analysis:
                        print(f"   • Recent Volume: {volume_analysis['recent_avg_volume']:,.0f}")
                        print(f"   • Volume Status: {'HIGH' if has_high_volume else 'NORMAL'}")
                
                print(f"\n{'='*60}")
                print(f"🎯 Market conditions not suitable for Fibonacci retracement analysis")
                print(f"{'='*60}")
        else:
            print("No intraday data available. Analyzing daily data...")
            fib_start, fib_end, start_time, end_time, move_pct, trend = detect_trend_and_swings(daily_clean)
            if fib_start is not None:
                print(f"\nDaily {trend.title()} Detected ({move_pct:.2f}% move)")
                print(f"  Start: ₹{fib_start:.2f} on {start_time.date()}")
                print(f"  End:   ₹{fib_end:.2f} on {end_time.date()}")
                plot_fibonacci_clean(daily_clean, fib_start, fib_end, start_time, end_time, move_pct, trend)
            else:
                print("No significant swings found in daily data either.")

except Exception as e:
    print(f"\nError: {str(e)}")
    print("Possible solutions:")
    print("- Check internet connection")
    print("- Verify the ticker symbol is correct")
    print("- Try reducing SWING_ANALYSIS_DAYS parameter")
    print("- Check if market is open (intraday data might be limited on weekends/holidays)")