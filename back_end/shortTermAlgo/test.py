import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# Configuration
TICKER = "SBIN.NS"  # SBI stock on NSE
INTRADAY_TIMEFRAME = "15m"  # 15-minute candles for intraday
SWING_ANALYSIS_DAYS = 15  # Look back 15 days for swing points
MIN_SWING_MOVE_PCT = 1.5  # Minimum price movement to qualify as a swing

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

def detect_swings(df, min_move_pct=MIN_SWING_MOVE_PCT):
    """Identify the most significant swing in the given dataframe"""
    # Check if dataframe is empty
    if df.empty:
        return None, None, None, None, None
    
    # Check if Volume column exists and filter out non-trading periods
    if 'Volume' in df.columns:
        try:
            # Handle NaN values and create volume mask safely
            volume_series = df['Volume'].fillna(0)  # Replace NaN with 0
            volume_mask = volume_series > 0
            
            # Check if any valid trading periods exist
            if not volume_mask.any():  # If no trading periods found
                return None, None, None, None, None
            
            df_filtered = df.loc[volume_mask].copy()
        except Exception as e:
            print(f"Volume filtering error: {e}")
            # If volume filtering fails, use all data
            df_filtered = df.copy()
    else:
        # If no Volume column, use all data
        df_filtered = df.copy()
    
    # Check if filtered dataframe is empty
    if df_filtered.empty:
        return None, None, None, None, None
    
    # Find highest high and lowest low
    swing_high = df_filtered['High'].max()
    swing_low = df_filtered['Low'].min()
    
    # Check for NaN values
    if pd.isna(swing_high) or pd.isna(swing_low):
        return None, None, None, None, None
    
    move_pct = ((swing_high - swing_low) / swing_low) * 100
    
    if move_pct >= min_move_pct:
        # Get the first occurrence of the swing high and low
        high_mask = df_filtered['High'] == swing_high
        low_mask = df_filtered['Low'] == swing_low
        
        # Check if any matches found
        if not high_mask.any() or not low_mask.any():
            return None, None, None, None, None
        
        high_time = df_filtered[high_mask].index[0]
        low_time = df_filtered[low_mask].index[0]
        
        return swing_high, swing_low, high_time, low_time, move_pct
    return None, None, None, None, None

def plot_fibonacci(df, swing_high, swing_low, high_time, low_time, move_pct):
    """Visualize the Fibonacci retracement levels"""
    # Calculate Fib levels
    fib_levels = {
        '0% (High)': swing_high,
        '23.6%': swing_high - (swing_high - swing_low) * 0.236,
        '38.2%': swing_high - (swing_high - swing_low) * 0.382,
        '50%': swing_high - (swing_high - swing_low) * 0.5,
        '61.8%': swing_high - (swing_high - swing_low) * 0.618,
        '78.6%': swing_high - (swing_high - swing_low) * 0.786,
        '100% (Low)': swing_low
    }
    
    # Plotting
    plt.figure(figsize=(15, 8))
    
    # Price data
    plt.plot(df.index, df['Close'], label='Price', color='navy', linewidth=1.5)
    
    # Swing markers
    plt.scatter(high_time, swing_high, color='red', s=100, label='Swing High', zorder=5)
    plt.scatter(low_time, swing_low, color='green', s=100, label='Swing Low', zorder=5)
    
    # Fibonacci levels
    colors = ['red', 'orange', 'gold', 'green', 'teal', 'purple', 'blue']
    for (level, price), color in zip(fib_levels.items(), colors):
        plt.axhline(price, color=color, linestyle='--', alpha=0.7, label=f'Fib {level}')
    
    # Formatting
    plt.title(f'SBI Fibonacci Retracement | {INTRADAY_TIMEFRAME} Data | {move_pct:.2f}% Move', pad=20)
    plt.xlabel('Time')
    plt.ylabel('Price (₹)')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print levels
    current_price = df['Close'].iloc[-1]
    print("\nFibonacci Retracement Levels:")
    for level, price in fib_levels.items():
        dist_pct = (current_price - price)/price * 100
        print(f"{level:<12}: ₹{price:.2f} | Distance from current: {dist_pct:+.2f}%")

# Main execution
try:
    # Check if it's weekend
    today = datetime.now()
    if today.weekday() >= 5:
        print(f"Note: Today is {'Saturday' if today.weekday() == 5 else 'Sunday'}. Indian markets are closed.")
        print("Using data up to last Friday for analysis.\n")
    
    print(f"Fetching {SWING_ANALYSIS_DAYS} days of {INTRADAY_TIMEFRAME} data for {TICKER}...")
    intraday_data, daily_data = fetch_stock_data()
    
    # Check if data was fetched successfully
    if intraday_data.empty and daily_data.empty:
        print("No data fetched. Please check your internet connection and ticker symbol.")
    else:
        print(f"Fetched {len(intraday_data)} intraday records and {len(daily_data)} daily records")
        
        # Detect swings in intraday data
        swing_high, swing_low, high_time, low_time, move_pct = detect_swings(intraday_data)
        
        if swing_high is not None and not pd.isna(swing_high):
            print(f"\nSignificant Intraday Swing Detected ({move_pct:.2f}% move)")
            print(f"  High: ₹{swing_high:.2f} at {high_time}")
            print(f"  Low:  ₹{swing_low:.2f} at {low_time}")
            plot_fibonacci(intraday_data, swing_high, swing_low, high_time, low_time, move_pct)
        else:
            print("\nNo significant swing detected in intraday data. Checking daily...")
            swing_high, swing_low, high_time, low_time, move_pct = detect_swings(daily_data)
            if swing_high is not None and not pd.isna(swing_high):
                print(f"\nDaily Swing Detected ({move_pct:.2f}% move)")
                print(f"  High: ₹{swing_high:.2f} on {high_time.date()}")
                print(f"  Low:  ₹{swing_low:.2f} on {low_time.date()}")
                plot_fibonacci(daily_data, swing_high, swing_low, high_time, low_time, move_pct)
            else:
                print("No tradable swings found in the given timeframe.")
                print("Try reducing MIN_SWING_MOVE_PCT or increasing SWING_ANALYSIS_DAYS")

except Exception as e:
    print(f"\nError: {str(e)}")
    print("Possible solutions:")
    print("- Check internet connection")
    print("- Verify the ticker symbol is correct")
    print("- Try reducing SWING_ANALYSIS_DAYS parameter")
    print("- Check if market is open (intraday data might be limited on weekends/holidays)")