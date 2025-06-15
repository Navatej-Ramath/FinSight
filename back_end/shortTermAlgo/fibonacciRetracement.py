import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def detect_complete_swings(df, min_move_percent=1.0):
    """
    Detect the most significant swing across the ENTIRE dataset
    Returns: (swing_high, swing_low, high_time, low_time)
    """
    # Find absolute high and low
    absolute_high = df['Price'].max()
    absolute_low = df['Price'].min()
    
    # Get their timestamps
    high_time = df[df['Price'] == absolute_high].index[0]
    low_time = df[df['Price'] == absolute_low].index[0]
    
    # Calculate total move percentage
    move_percent = ((absolute_high - absolute_low) / absolute_low) * 100
    
    # Only validate if meets minimum move requirement
    if move_percent >= min_move_percent:
        return absolute_high, absolute_low, high_time, low_time, move_percent
    else:
        return None, None, None, None, None

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
    
    # Detect the most significant swing in entire dataset
    swing_high, swing_low, high_time, low_time, move_percent = detect_complete_swings(df)
    
    if swing_high is None:
        print("\nNo significant swing detected (minimum move requirement not met)")
        
        # Plot just the price
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['Price'], label='Price', color='royalblue')
        plt.title('Price Chart - No Significant Swing', pad=20)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        # Determine swing direction
        if high_time > low_time:
            direction = "Uptrend"
            start, end = low_time, high_time
        else:
            direction = "Downtrend"
            start, end = high_time, low_time
        
        print(f"\nSignificant {direction} detected:")
        print(f"  Swing High: {swing_high:.2f} at {high_time}")
        print(f"  Swing Low:  {swing_low:.2f} at {low_time}")
        print(f"  Total Move: {move_percent:.2f}%")
        print(f"  Duration:   {(end - start).total_seconds()/60:.1f} minutes")
        
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
        
        # Visualization
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['Price'], label='Price', color='royalblue', linewidth=1.5)
        
        # Highlight swing period
        plt.axvspan(start, end, alpha=0.1, color='green', label=f'{direction} Period')
        
        # Mark swing points
        plt.scatter(high_time, swing_high, color='red', s=100, label='Swing High')
        plt.scatter(low_time, swing_low, color='green', s=100, label='Swing Low')
        
        # Plot Fibonacci levels
        colors = ['red', 'orange', 'gold', 'green', 'teal', 'purple', 'brown']
        for (level, price), color in zip(fib_levels.items(), colors):
            plt.axhline(price, color=color, linestyle='--', alpha=0.7, label=f'{level}: {price:.2f}')
        
        plt.title(f'Complete Swing Analysis | {direction} {move_percent:.2f}%', pad=20)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Text output
        print("\n=== FIBONACCI LEVELS ===")
        for level, price in fib_levels.items():
            dist_pct = (current_price - price)/price * 100
            print(f"{level:<10}: {price:.2f} | Current: {dist_pct:+.2f}% {'above' if dist_pct > 0 else 'below'}")
        
        plt.show()

except Exception as e:
    print(f"\nError: {str(e)}")
    print("Please check your CSV format. Expected columns:")
    print("1. DateTime | 2. Any | 3. Price")