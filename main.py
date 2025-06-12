import pandas as pd
import matplotlib.pyplot as plt

# Load CSV and automatically detect price column
try:
    df = pd.read_csv('chart.csv')
    price_col = df.columns[2]  # 3rd column (index 2)
    print(f"Detected price column: '{price_col}'")
    
    # Convert and clean data
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    df['Price'] = df[price_col].ffill().dropna()  # Use 3rd column and clean
    
    if len(df) < 30:
        raise ValueError("Not enough data points (need at least 30)")
    
    # Swing point detection
    window = max(15, len(df)//10)  # Adaptive window size
    df['SwingHigh'] = df['Price'].rolling(window).max()
    df['SwingLow'] = df['Price'].rolling(window).min()

    # Get valid swings
    valid_highs = df['SwingHigh'].dropna()
    valid_lows = df['SwingLow'].dropna()
    
    if len(valid_highs) < 2 or len(valid_lows) < 2:
        raise ValueError("Insufficient swing points detected")

    swing_high = valid_highs.iloc[-1]
    swing_low = valid_lows.iloc[-1]

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
    
    # Plot Fibonacci levels
    colors = ['red', 'orange', 'gold', 'green', 'teal', 'purple', 'brown']
    for (level, price), color in zip(fib_levels.items(), colors):
        plt.axhline(price, color=color, linestyle='--', alpha=0.7, label=f'{level}: {price:.2f}')
    
    plt.title(f'Intraday Fibonacci Retracement\n{df.index[0].date()} | Current: {current_price:.2f}', pad=20)
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