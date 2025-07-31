"""
Configuration file for the Technical Analysis Suite
"""

APP_CONFIG = {
    'app_name': 'Technical Analysis Suite',
    'version': '1.0.0',
    'author': 'Technical Analysis Team',
    'description': 'Advanced technical analysis tools for Indian stock market',
    'data_source': 'Yahoo Finance API',
    'supported_exchanges': ['NSE', 'BSE'],
    'default_currency': 'INR',
    'cache_duration': 300,  # 5 minutes
    'max_lookback_days': 365,
    'min_lookback_days': 1
}

# Fibonacci specific configuration
FIBONACCI_CONFIG = {
    'default_min_swing_pct': 1.5,
    'default_scan_window_minutes': 90,
    'default_volume_multiplier': 1.5,
    'fibonacci_levels': [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0],
    'level_colors': ['red', 'orange', 'gold', 'green', 'teal', 'purple', 'blue'],
    'chart_height': 800,
    'default_timeframe': '15m'
}

# Data fetching configuration
DATA_CONFIG = {
    'default_ticker': 'SBIN.NS',
    'yahoo_finance_timeout': 30,
    'max_retries': 3,
    'supported_timeframes': {
        '1m': '1 Minute',
        '2m': '2 Minutes', 
        '5m': '5 Minutes',
        '15m': '15 Minutes',
        '30m': '30 Minutes',
        '60m': '1 Hour',
        '90m': '90 Minutes',
        '1h': '1 Hour',
        '1d': '1 Day',
        '5d': '5 Days',
        '1wk': '1 Week',
        '1mo': '1 Month',
        '3mo': '3 Months'
    }
}

# UI Configuration
UI_CONFIG = {
    'page_title': 'Technical Analysis Suite',
    'page_icon': '📈', 
    'layout': 'wide',
    'sidebar_state': 'expanded',
    'theme': 'light',
    'primary_color': '#1e3c72',
    'secondary_color': '#2a5298',
    'success_color': '#28a745',
    'warning_color': '#ffc107',
    'error_color': '#dc3545',
    'info_color': '#17a2b8'
}

# Market configuration
MARKET_CONFIG = {
    'trading_hours': {
        'start': '09:15',
        'end': '15:30',
        'timezone': 'Asia/Kolkata'
    },
    'trading_days': [0, 1, 2, 3, 4],  # Monday to Friday
    'holidays': [],  # Can be populated with market holidays
    'pre_market_start': '09:00',
    'after_market_end': '16:00'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'technical_analysis.log',
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# Feature flags
FEATURE_FLAGS = {
    'enable_caching': True,
    'enable_logging': True,
    'enable_error_reporting': True,
    'enable_performance_monitoring': True,
    'enable_advanced_charts': True,
    'enable_export_features': True,
    'enable_alerts': False,  # Future feature
    'enable_portfolio_tracking': False,  # Future feature
    'enable_backtesting': False  # Future feature
}

# Algorithm-specific configurations
ALGORITHM_CONFIGS = {
    'fibonacci': FIBONACCI_CONFIG,
    'rsi': {
        'default_period': 14,
        'overbought_level': 70,
        'oversold_level': 30,
        'divergence_lookback': 20
    },
    'bollinger_bands': {
        'default_period': 20,
        'default_std_dev': 2,
        'squeeze_threshold': 0.1
    },
    'macd': {
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9
    }
}