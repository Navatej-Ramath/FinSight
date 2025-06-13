import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --- CONFIGURATION ---
SWING_WINDOW = 30        # Number of candles to look back for swing detection
MIN_SWING_PCT = 0.4      # Minimum price movement (%) to qualify as a swing
SMOOTHING_PERIOD = 5     # EMA period for noise reduction
FIB_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]  # Standard Fibonacci levels

# --- DATA LOADING & PREPROCESSING ---
def load_data(filepath):
    """Load and preprocess the CSV file"""
    df = pd.read_csv(filepath)
    
    # Auto-detect datetime and price columns
    datetime_col = None
    price_col = None
    
    # Find datetime column (look for 'time', 'date' etc)
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['time', 'date', 'datetime']):
            datetime_col = col
            break
    if datetime_col is None:
        datetime_col = df.columns[0]  # Fallback to first column
    
    # Find price column (look for 'price', 'close' etc)
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['price', 'close', 'last', 'value']):
            price_col = col
            break
    if price_col is None:
        price_col = df.columns[-1]  # Fallback to last column
    
    print(f"Detected columns - DateTime: '{datetime_col}', Price: '{price_col}'")
    
    # Convert and clean data
    df['DateTime'] = pd.to_datetime(df[datetime_col])
    df.set_index('DateTime', inplace=True)
    df['Price'] = pd.to_numeric(df[price_col], errors='coerce').ffill()
    
    return df

# --- SWING DETECTION ---
def detect_swings(price_series, window=SWING_WINDOW, min_pct=MIN_SWING_PCT):
    """Identify significant swing highs and lows"""
    highs, lows = [], []
    
    for i in range(window, len(price_series)-window):
        current = price_series.iloc[i]
        lookback = price_series.iloc[i-window:i+window]
        
        # Check for swing high
        if current == lookback.max():
            move_pct = (current - lookback.min()) / lookback.min() * 100
            if move_pct >= min_pct:
                highs.append((price_series.index[i], current))
        
        # Check for swing low
        elif current == lookback.min():
            move_pct = (lookback.max() - current) / current * 100
            if move_pct >= min_pct:
                lows.append((price_series.index[i], current))
    
    return pd.DataFrame(highs, columns=['Time', 'High']), pd.DataFrame(lows, columns=['Time', 'Low'])

# --- FIBONACCI CALCULATIONS ---
def calculate_fib_levels(swing_high, swing_low):
    """Calculate Fibonacci retracement levels"""
    price_diff = swing_high - swing_low
    return {f"{level*100:.1f}%": swing_high - price_diff * level for level in FIB_LEVELS}

# --- VISUALIZATION ---
def plot_fibonacci(df, swing_high, swing_low, fib_levels):
    """Create the Fibonacci retracement plot"""
    plt.figure(figsize=(15, 7))
    
    # Plot price data
    plt.plot(df.index, df['Price'], color='dodgerblue', alpha=0.4, label='Price')
    plt.plot(df.index, df['Smoothed'], color='navy', linewidth=1.5, label='Smoothed')
    
    # Mark swing points
    plt.scatter(swing_high['Time'], swing_high['High'], color='red', s=100, label='Swing High')
    plt.scatter(swing_low['Time'], swing_low['Low'], color='green', s=100, label='Swing Low')
    
    # Draw Fibonacci levels
    for level, price in fib_levels.items():
        plt.axhline(y=price, linestyle='--', alpha=0.7, 
                   label=f'Fib {level} ({price:.2f})')
    
    # Formatting
    plt.title('Fibonacci Retracement Analysis', pad=20)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(15))  # Limit x-axis ticks
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION ---
def main():
    try:
        # Load and prepare data
        df = load_data('chart.csv')
        df['Smoothed'] = df['Price'].ewm(span=SMOOTHING_PERIOD).mean()
        
        # Detect swings
        swing_highs, swing_lows = detect_swings(df['Smoothed'])
        
        if len(swing_highs) == 0 or len(swing_lows) == 0:
            raise ValueError("No valid swings detected. Try adjusting SWING_WINDOW or MIN_SWING_PCT")
        
        # Get most recent swing high/low
        last_high = swing_highs.iloc[-1]
        last_low = swing_lows.iloc[-1]
        
        # Determine swing direction
        if last_high['Time'] > last_low['Time']:
            start, end = last_low, last_high
            direction = "Uptrend"
        else:
            start, end = last_high, last_low
            direction = "Downtrend"
        
        # Calculate Fibonacci levels
        fib_levels = calculate_fib_levels(end['High' if direction == "Uptrend" else 'Low'], 
                                         start['Low' if direction == "Uptrend" else 'High'])
        
        # Generate output
        current_price = df['Price'].iloc[-1]
        print(f"\nMarket Direction: {direction}")
        print(f"Swing Points: {start['Time']} ({start['Low' if direction == 'Uptrend' else 'High']:.2f}) → {end['Time']} ({end['High' if direction == 'Uptrend' else 'Low']:.2f})")
        print("\nFibonacci Levels:")
        for level, price in fib_levels.items():
            dist_pct = (current_price - price) / price * 100
            print(f"{level:>6}: {price:.2f} | Current: {dist_pct:+.2f}% {'ABOVE' if dist_pct > 0 else 'BELOW'}")
        
        # Plot results
        plot_fibonacci(df, end if direction == "Uptrend" else start, 
                      start if direction == "Uptrend" else end, fib_levels)
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Possible solutions:")
        print("- Check your CSV file format (expected columns: DateTime, Price)")
        print("- Adjust SWING_WINDOW or MIN_SWING_PCT parameters")
        print("- Ensure you have sufficient data points (at least 100 candles recommended)")

if __name__ == "__main__":
    main()