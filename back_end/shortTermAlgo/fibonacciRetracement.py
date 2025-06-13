import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def detect_valid_swings(df, window_size=12, min_move_percent=0.8):  # window_size=12 for 60min (12*5min)
    """
    Detect valid swings using rolling windows
    Returns: (found_swing, swing_high, swing_low, swing_start, swing_end) or (False, None, None, None, None)
    """
    # Calculate percentage moves and direction
    df['pct_move'] = df['Price'].pct_change() * 100
    df['direction'] = np.sign(df['pct_move'])
    
    # Initialize swing tracking
    best_swing = None
    max_quality = 0
    
    # Slide through the data with rolling window
    for i in range(len(df) - window_size):
        window = df.iloc[i:i+window_size]
        window_high = window['Price'].max()
        window_low = window['Price'].min()
        move_percent = ((window_high - window_low) / window_low) * 100
        
        # Only consider significant moves
        if move_percent >= min_move_percent:
            # Calculate direction consistency
            direction_changes = window['direction'].diff().abs().sum()
            
            # Quality score (higher is better)
            quality_score = move_percent * (1 - (direction_changes / window_size))
            
            # Track the best swing (highest quality score)
            if quality_score > max_quality:
                max_quality = quality_score
                best_swing = {
                    'high': window_high,
                    'low': window_low,
                    'start': window.index[0],
                    'end': window.index[-1],
                    'quality': quality_score
                }
    
    if best_swing and best_swing['quality'] > (min_move_percent * 0.7):
        return True, best_swing['high'], best_swing['low'], best_swing['start'], best_swing['end']
    
    return False, None, None, None, None

try:
    # Load and prepare data
    df = pd.read_csv('chart.csv')
    price_col = df.columns[2]  # 3rd column (index 2)
    print(f"Detected price column: '{price_col}'")
    
    # Convert and clean data
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    df['Price'] = df[price_col].ffill().dropna()
    
    if len(df) < 30:
        raise ValueError("Not enough data points (need at least 30)")
    
    # Resample to 5-minute candles using 'min' instead of deprecated 'T'
    df_resampled = df['Price'].resample('5min').last().ffill().to_frame()
    
    # Detect valid swings (using 12 periods = 60 minutes)
    has_swing, swing_high, swing_low, swing_start, swing_end = detect_valid_swings(df_resampled)
    
    if not has_swing:
        print("\nNo valid swing detected - Fibonacci retracement not applicable")
        print("Possible reasons:")
        print("- Price movement too small (<1.5%)")
        print("- Too many direction changes (whipsaws)")
        print("- No clear trend in the last 60 minutes")
        
        # Plot just the price
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['Price'], label='Price', color='royalblue')
        plt.title('Price Chart - No Valid Swing Detected', pad=20)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print(f"\nValid swing detected ({swing_start} to {swing_end}):")
        print(f"High: {swing_high:.2f}, Low: {swing_low:.2f}")
        print(f"Move: {((swing_high - swing_low)/swing_low)*100:.2f}%")
        
        # Fibonacci calculations
        def get_fib_levels(h, l):
            diff = h - l
            return {
                '0% (High)': h,
                '23.6%': h - diff * 0.236,
                '38.2%': h - diff * 0.382,
                '50%': h - diff * 0.5,
                '61.8%': h - diff * 0.618,
                '78.6%': h - diff * 0.786,
                '100% (Low)': l
            }

        fib_levels = get_fib_levels(swing_high, swing_low)
        current_price = df['Price'].iloc[-1]
        
        # Trade detection
        signals = []
        for level, price in fib_levels.items():
            threshold = 0.005 * (swing_high - swing_low)  # Dynamic threshold
            if abs(current_price - price) <= threshold:
                trend = "Up" if df['Price'].iloc[-10:].mean() > df['Price'].iloc[-20:-10].mean() else "Down"
                action = 'BUY' if ('%' in level and float(level.split('%')[0]) > 38) and trend == "Up" else 'SELL'
                signals.append({
                    'Level': level,
                    'Price': round(price, 2),
                    'Action': action,
                    'Distance (%)': round(abs(current_price - price)/price*100, 2)
                })

        # Visualization
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['Price'], label='Price', color='royalblue', linewidth=1.5)
        
        # Highlight the swing area
        plt.axvspan(swing_start, swing_end, alpha=0.1, color='green', label='Swing Period')
        plt.axhline(swing_high, color='red', linestyle=':', alpha=0.5)
        plt.axhline(swing_low, color='blue', linestyle=':', alpha=0.5)
        
        # Plot Fibonacci levels
        colors = ['red', 'orange', 'gold', 'green', 'teal', 'purple', 'brown']
        for (level, price), color in zip(fib_levels.items(), colors):
            plt.axhline(price, color=color, linestyle='--', alpha=0.7, label=f'{level}: {price:.2f}')
        
        plt.title(f'Fibonacci Retracement\n{swing_start} to {swing_end} | Current: {current_price:.2f}', pad=20)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Text output
        print("\n=== FIBONACCI LEVELS ===")
        for level, price in fib_levels.items():
            print(f"{level:<10}: {price:.2f}")
        
        if signals:
            print("\n=== TRADE SIGNALS ===")
            for sig in signals:
                print(f"{sig['Action']} near {sig['Level']} ({sig['Price']}) | {sig['Distance (%)']}% away")
        else:
            print("\nNo trade signals detected at current price")
        
        plt.show()

except Exception as e:
    print(f"\nError: {str(e)}")
    print("Please check your CSV format. Expected columns:")
    print("1. DateTime | 2. Any | 3. Price")