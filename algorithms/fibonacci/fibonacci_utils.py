"""
Fibonacci Retracement Utilities
Helper functions and plotting utilities for Fibonacci analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, Any
from datetime import datetime

def create_plotly_fibonacci_chart(df: pd.DataFrame, fib_start: float, fib_end: float, 
                                start_time: datetime, end_time: datetime, 
                                move_pct: float, trend: str, ticker: str,
                                timeframe: str, volume_analysis: Optional[Dict] = None,
                                is_window_analysis: bool = False) -> go.Figure:
    """
    Create interactive Plotly chart with Fibonacci levels
    
    Args:
        df: DataFrame with OHLCV data
        fib_start: Starting price of swing
        fib_end: Ending price of swing
        start_time: Start time of swing
        end_time: End time of swing
        move_pct: Movement percentage
        trend: Trend direction
        ticker: Stock ticker symbol
        timeframe: Chart timeframe
        volume_analysis: Volume analysis results
        is_window_analysis: Whether this is window-based analysis
        
    Returns:
        Plotly figure object
    """
    # Find columns
    close_col = find_column(df, 'Close')
    high_col = find_column(df, 'High')
    low_col = find_column(df, 'Low')
    open_col = find_column(df, 'Open')
    volume_col = find_column(df, 'Volume')
    
    # Calculate Fibonacci levels
    fib_levels = calculate_fibonacci_levels(fib_start, fib_end, trend)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price Chart with Fibonacci Levels', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Price chart - OHLC/Candlestick
    if open_col and all([open_col, high_col, low_col, close_col]):
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df[open_col],
                high=df[high_col],
                low=df[low_col],
                close=df[close_col],
                name='Price',
                showlegend=False
            ),
            row=1, col=1
        )
    else:
        # Fallback to line chart if OHLC not available
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[close_col],
                mode='lines',
                name='Price',
                line=dict(color='navy', width=2),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Swing points
    fig.add_trace(
        go.Scatter(
            x=[start_time],
            y=[fib_start],
            mode='markers',
            marker=dict(color='blue', size=15, symbol='circle'),
            name=f'Swing Start ({trend.title()})',
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[end_time],
            y=[fib_end],
            mode='markers',
            marker=dict(color='red', size=15, symbol='circle'),
            name=f'Swing End ({trend.title()})',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Fibonacci trend line
    fig.add_trace(
        go.Scatter(
            x=[start_time, end_time],
            y=[fib_start, fib_end],
            mode='lines',
            line=dict(color='black', width=3),
            name='Fibonacci Trend Line',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Fibonacci levels
    colors = ['red', 'orange', 'gold', 'green', 'teal', 'purple', 'blue']
    for (level, price), color in zip(fib_levels.items(), colors):
        fig.add_hline(
            y=price,
            line=dict(color=color, dash='dash', width=2),
            annotation_text=f'Fib {level}: ₹{price:.2f}',
            annotation_position="right",
            row=1
        )
    
    # Volume chart
    if volume_col is not None:
        # Color volume bars based on price movement
        if open_col:
            colors_vol = ['red' if close < open else 'green' 
                         for close, open in zip(df[close_col], df[open_col])]
        else:
            colors_vol = ['lightblue'] * len(df)
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df[volume_col],
                name='Volume',
                marker_color=colors_vol,
                showlegend=False,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # High volume threshold
        if volume_analysis and 'high_volume_threshold' in volume_analysis:
            fig.add_hline(
                y=volume_analysis['high_volume_threshold'],
                line=dict(color='red', dash='dot', width=2),
                annotation_text=f"High Vol Threshold: {volume_analysis['high_volume_threshold']:,.0f}",
                annotation_position="right",
                row=2
            )
    
    # Update layout
    trend_arrow = "↗" if trend == "uptrend" else "↘"
    window_text = f" (Window Analysis)" if is_window_analysis else ""
    
    fig.update_layout(
        title=f'{ticker} Fibonacci Retracement | {trend.title()} {trend_arrow} | {timeframe} | {move_pct:.2f}% Move{window_text}',
        xaxis_title='Time',
        yaxis_title='Price (₹)',
        yaxis2_title='Volume',
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )
    
    return fig

def create_matplotlib_fibonacci_chart(df: pd.DataFrame, fib_start: float, fib_end: float,
                                     start_time: datetime, end_time: datetime,
                                     move_pct: float, trend: str, ticker: str,
                                     timeframe: str, volume_analysis: Optional[Dict] = None,
                                     is_window_analysis: bool = False) -> plt.Figure:
    """
    Create matplotlib chart with Fibonacci levels (fallback option)
    
    Args:
        df: DataFrame with OHLCV data
        fib_start: Starting price of swing
        fib_end: Ending price of swing
        start_time: Start time of swing
        end_time: End time of swing
        move_pct: Movement percentage
        trend: Trend direction
        ticker: Stock ticker symbol
        timeframe: Chart timeframe
        volume_analysis: Volume analysis results
        is_window_analysis: Whether this is window-based analysis
        
    Returns:
        Matplotlib figure object
    """
    # Find columns
    close_col = find_column(df, 'Close')
    high_col = find_column(df, 'High')
    low_col = find_column(df, 'Low')
    volume_col = find_column(df, 'Volume')
    
    # Calculate Fibonacci levels
    fib_levels = calculate_fibonacci_levels(fib_start, fib_end, trend)
    
    # Create clean time series for plotting (no gaps)
    df_clean = df.copy()
    df_clean['plot_x'] = range(len(df_clean))
    
    # Find plot positions for swing points
    start_plot_x = df_clean[df_clean.index == start_time]['plot_x'].iloc[0]
    end_plot_x = df_clean[df_clean.index == end_time]['plot_x'].iloc[0]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[3, 1])
    
    # Price chart
    ax1.plot(df_clean['plot_x'], df_clean[close_col], 
             label='Close Price', color='navy', linewidth=1.5)
    
    if high_col and low_col:
        ax1.plot(df_clean['plot_x'], df_clean[low_col], 
                 color='red', linewidth=1, alpha=0.7)
        ax1.plot(df_clean['plot_x'], df_clean[high_col], 
                 color='green', linewidth=1, alpha=0.7)
    
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
    window_text = f" (Window)" if is_window_analysis else ""
    title = f'{ticker} Fibonacci Retracement | {trend.title()} {trend_arrow} | {timeframe} | {move_pct:.2f}% Move{window_text}'
    
    ax1.set_title(title, pad=20, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (₹)', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax1.grid(alpha=0.3)
    
    ax2.set_xlabel('Time (No Gaps)', fontsize=12)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def calculate_fibonacci_levels(fib_start: float, fib_end: float, trend: str) -> Dict[str, float]:
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

def find_column(df: pd.DataFrame, column_type: str) -> Optional[str]:
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

def format_levels_for_display(fib_levels: Dict[str, float], current_price: float) -> pd.DataFrame:
    """
    Format Fibonacci levels for display in tables
    
    Args:
        fib_levels: Dictionary of Fibonacci levels
        current_price: Current market price
        
    Returns:
        Formatted DataFrame for display
    """
    levels_data = []
    for level, price in fib_levels.items():
        if price > 0:
            dist_pct = abs(current_price - price) / price * 100
            position = "Above" if current_price > price else "Below"
            levels_data.append({
                "Fibonacci Level": level,
                "Price (₹)": f"₹{price:.2f}",
                "Current Position": f"{position} by {dist_pct:.2f}%",
                "Distance": dist_pct
            })
    
    return pd.DataFrame(levels_data)

def print_analysis_summary(fib_start: float, fib_end: float, start_time: datetime,
                          end_time: datetime, move_pct: float, trend: str,
                          current_price: float, fib_levels: Dict[str, float],
                          volume_analysis: Optional[Dict] = None, 
                          is_window_analysis: bool = False) -> None:
    """
    Print comprehensive analysis summary to console
    
    Args:
        fib_start: Starting price of swing
        fib_end: Ending price of swing
        start_time: Start time of swing
        end_time: End time of swing
        move_pct: Movement percentage
        trend: Trend direction
        current_price: Current market price
        fib_levels: Dictionary of Fibonacci levels
        volume_analysis: Volume analysis results
        is_window_analysis: Whether this is window-based analysis
    """
    analysis_type = "WINDOW" if is_window_analysis else "FULL DATASET"
    trend_arrow = "↗" if trend == "uptrend" else "↘"
    
    print(f"\n{trend.upper()} Fibonacci Retracement Analysis ({analysis_type}):")
    print(f"Trend Direction: {trend.title()} {trend_arrow}")
    print(f"Current Price: ₹{current_price:.2f}")
    print(f"Swing Move: {move_pct:.2f}% (₹{fib_start:.2f} → ₹{fib_end:.2f})")
    print(f"Time Period: {start_time} → {end_time}")
    
    if volume_analysis:
        print(f"\nVolume Analysis:")
        print(f"Overall Avg Volume: {volume_analysis['avg_volume_overall']:,.0f}")
        print(f"Recent Avg Volume: {volume_analysis['recent_avg_volume']:,.0f}")
        print(f"Volume Ratio: {volume_analysis['volume_ratio']:.2f}x")
        print(f"High Volume Periods: {volume_analysis['high_volume_periods']}")
    
    print(f"\nFibonacci Levels:")
    for level, price in fib_levels.items():
        dist_pct = abs(current_price - price)/price * 100 if price > 0 else 0
        position = "Above" if current_price > price else "Below"
        print(f"{level:<18}: ₹{price:.2f} | {position} by {dist_pct:.2f}%")

def get_trading_recommendations(trend: str, current_price: float, 
                              fib_levels: Dict[str, float]) -> Dict[str, Any]:
    """
    Generate trading recommendations based on Fibonacci analysis
    
    Args:
        trend: Trend direction
        current_price: Current market price
        fib_levels: Dictionary of Fibonacci levels
        
    Returns:
        Dictionary of trading recommendations
    """
    # Find key levels
    support_level = min(fib_levels.values())
    resistance_level = max(fib_levels.values())
    fifty_percent = fib_levels['50%']
    golden_ratio = fib_levels['61.8%']
    
    # Determine current position
    levels_above = [price for price in fib_levels.values() if price > current_price]
    levels_below = [price for price in fib_levels.values() if price < current_price]
    
    recommendations = {
        'trend_direction': trend,
        'key_levels': {
            'support': support_level,
            'resistance': resistance_level,
            'fifty_percent': fifty_percent,
            'golden_ratio': golden_ratio
        },
        'current_position': 'neutral'
    }
    
    if trend == "uptrend":
        if current_price > fifty_percent:
            recommendations['current_position'] = 'strong_uptrend'
            recommendations['action'] = 'Look for buying opportunities on dips to 50% or 61.8% levels'
            recommendations['stop_loss'] = f"Below ₹{golden_ratio:.2f} (61.8% level)"
            recommendations['target'] = f"₹{resistance_level:.2f} (100% level)"
        else:
            recommendations['current_position'] = 'retracement_phase'
            recommendations['action'] = 'Wait for bounce from support levels before buying'
            recommendations['stop_loss'] = f"Below ₹{support_level:.2f} (0% level)"
            recommendations['target'] = f"₹{fifty_percent:.2f} (50% level)"
    
    else:  # downtrend
        if current_price < fifty_percent:
            recommendations['current_position'] = 'strong_downtrend'
            recommendations['action'] = 'Look for selling opportunities on rallies to 50% or 61.8% levels'
            recommendations['stop_loss'] = f"Above ₹{golden_ratio:.2f} (61.8% level)"
            recommendations['target'] = f"₹{support_level:.2f} (100% level)"
        else:
            recommendations['current_position'] = 'retracement_phase'
            recommendations['action'] = 'Wait for rejection from resistance levels before selling'
            recommendations['stop_loss'] = f"Above ₹{resistance_level:.2f} (0% level)"
            recommendations['target'] = f"₹{fifty_percent:.2f} (50% level)"
    
    # Add general recommendations
    recommendations['general_tips'] = [
        "Use proper position sizing and risk management",
        "Confirm signals with volume analysis",
        "Consider market conditions and news events",
        "Fibonacci levels work best in trending markets",
        "Use multiple timeframes for confirmation"
    ]
    
    return recommendations