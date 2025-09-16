"""
Data Fetcher Module
Handles fetching and caching of stock data from various sources
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
import streamlit as st

class DataFetcher:
    """
    Class to handle data fetching from Yahoo Finance and other sources
    """
    
    def __init__(self):
        self.cache_duration = 300  # 5 minutes cache
    
    @st.cache_data(ttl=300)
    def fetch_stock_data(_self, ticker: str, lookback_days: int, 
                        timeframe: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch historical data from Yahoo Finance with caching
        
        Args:
            ticker: Stock ticker symbol
            lookback_days: Number of days to look back
            timeframe: Chart timeframe (15m, 1h, 1d)
            
        Returns:
            Tuple of (DataFrame, error_message)
        """
        try:
            end_date = datetime.now()
            
            # If it's weekend, use last Friday as end date
            if end_date.weekday() >= 5:  # Saturday=5, Sunday=6
                days_back = end_date.weekday() - 4  # Go back to Friday
                end_date = end_date - timedelta(days=days_back)
            
            start_date = end_date - timedelta(days=lookback_days)
            
            # Fetch data with auto_adjust explicitly set to avoid warnings
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=timeframe,
                progress=False,
                auto_adjust=True,
                prepost=True,  # Include pre and post market data
                threads=True   # Use threading for faster downloads
            )
            
            if data.empty:
                return None, f"No data found for ticker {ticker}"
            
            return data, None
            
        except Exception as e:
            error_msg = f"Error fetching data for {ticker}: {str(e)}"
            return None, error_msg

    def get_financial_ratios(self, ticker_symbol: str) -> Dict[str, Any]:
        """
        Fetches and calculates key financial ratios for a given stock ticker from Yahoo Finance.
        """
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            if not info or info.get('trailingEps') is None:
                return {"Error": f"Could not retrieve valid data for {ticker_symbol}. It might be delisted or an invalid ticker."}

            financials = stock.financials
            balance_sheet = stock.balance_sheet
            financials_T = financials.T
            balance_sheet_T = balance_sheet.T

            asset_turnover = 'Not available'
            if not financials_T.empty and not balance_sheet_T.empty and len(balance_sheet_T) > 1:
                try:
                    total_revenue = financials_T['Total Revenue'].iloc[0]
                    avg_total_assets = (balance_sheet_T['Total Assets'].iloc[0] + balance_sheet_T['Total Assets'].iloc[1]) / 2
                    if avg_total_assets != 0:
                        asset_turnover = total_revenue / avg_total_assets
                except (KeyError, IndexError, TypeError):
                    pass

            inventory_turnover = 'Not available'
            if not financials_T.empty and not balance_sheet_T.empty and len(balance_sheet_T) > 1:
                try:
                    cost_of_goods_sold = financials_T['Cost Of Revenue'].iloc[0]
                    avg_inventory = (balance_sheet_T['Inventory'].iloc[0] + balance_sheet_T['Inventory'].iloc[1]) / 2
                    if avg_inventory is not None and avg_inventory != 0:
                        inventory_turnover = cost_of_goods_sold / avg_inventory
                    elif avg_inventory == 0:
                        inventory_turnover = 'Not applicable (zero inventory)'
                except (KeyError, IndexError, TypeError):
                    pass

            ratios = {
                "Ticker": ticker_symbol,
                "Net Margin": info.get('profitMargins'),
                "ROE": info.get('returnOnEquity'),
                "Current Ratio": info.get('currentRatio'),
                "Quick Ratio": info.get('quickRatio'),
                "Debt/Equity": info.get('debtToEquity'),
                "P/E Ratio": info.get('trailingPE'),
                "PEG Ratio": info.get('pegRatio'),
                "Payout Ratio": info.get('payoutRatio'),
                "Asset Turnover": asset_turnover,
                "Inventory Turnover": inventory_turnover,
            }
            return ratios

        except Exception as e:
            return {"Error": f"Could not retrieve data for {ticker_symbol}. Please check the ticker. Error: {e}"}

    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get basic stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'company_name': info.get('longName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('currentPrice', 0),
                'currency': info.get('currency', 'INR'),
                'exchange': info.get('exchange', 'NSE')
            }
        except:
            return {
                'company_name': ticker, 'sector': 'N/A', 'industry': 'N/A',
                'market_cap': 0, 'current_price': 0, 'currency': 'INR', 'exchange': 'NSE'
            }
    
    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get basic stock information
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'company_name': info.get('longName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('currentPrice', 0),
                'currency': info.get('currency', 'INR'),
                'exchange': info.get('exchange', 'NSE')
            }
        except:
            return {
                'company_name': ticker,
                'sector': 'N/A',
                'industry': 'N/A',
                'market_cap': 0,
                'current_price': 0,
                'currency': 'INR',
                'exchange': 'NSE'
            }
    def fetch_stock_data_and_info(self, ticker: str, period: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """Fetch stock data and company information"""
        try:
            # Fetch stock data
            stock = yf.Ticker(ticker)
            stock_data = stock.history(period=period)

            if stock_data.empty:
                return None, {}

            # Reset index to make Date a column
            stock_data.reset_index(inplace=True)

            # Fetch company info
            try:
                company_info = stock.info
            except:
                company_info = {}

            return stock_data, company_info

        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None, {}
    
    def validate_ticker(self, ticker: str) -> bool:
        """
        Validate if ticker exists and has data
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if valid, False otherwise
        """
        try:
            stock = yf.Ticker(ticker)
            # Try to get last 5 days of data
            data = stock.history(period="5d")
            return not data.empty
        except:
            return False
    
    def get_popular_nse_stocks(self) -> Dict[str, str]:
        
        return {
            'SBIN.NS': 'State Bank of India',
            'RELIANCE.NS': 'Reliance Industries',
            'TCS.NS': 'Tata Consultancy Services',
            'INFY.NS': 'Infosys Limited',
            'HDFCBANK.NS': 'HDFC Bank',
            'ICICIBANK.NS': 'ICICI Bank',
            'ITC.NS': 'ITC Limited',
            'WIPRO.NS': 'Wipro Limited',
            'LT.NS': 'Larsen & Toubro',
            'MARUTI.NS': 'Maruti Suzuki',
            'BAJFINANCE.NS': 'Bajaj Finance',
            'BHARTIARTL.NS': 'Bharti Airtel',
            'ASIANPAINT.NS': 'Asian Paints',
            'NESTLEIND.NS': 'Nestle India',
            'HINDUNILVR.NS': 'Hindustan Unilever',
            'KOTAKBANK.NS': 'Kotak Mahindra Bank',
            'AXISBANK.NS': 'Axis Bank',
            'ULTRACEMCO.NS': 'UltraTech Cement',
            'SUNPHARMA.NS': 'Sun Pharmaceutical',
            'TITAN.NS': 'Titan Company'
        }
    
    def get_timeframe_options(self) -> Dict[str, str]:
        """
        Get available timeframe options
        
        Returns:
            Dictionary of {timeframe: description}
        """
        return {
            '5m': '5 Minutes (Intraday)',
            '15m': '15 Minutes (Intraday)', 
            '30m': '30 Minutes (Intraday)',
            '1h': '1 Hour (Swing)',
            '2h': '2 Hours (Swing)',
            '1d': '1 Day (Position)',
            '5d': '5 Days (Weekly)',
            '1wk': '1 Week (Long-term)',
            '1mo': '1 Month (Long-term)'
        }
    
    def format_market_cap(self, market_cap: int) -> str:
        """
        Format market cap in readable format
        
        Args:
            market_cap: Market cap value
            
        Returns:
            Formatted string
        """
        if market_cap >= 1e12:
            return f"₹{market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            return f"₹{market_cap/1e9:.2f}B"
        elif market_cap >= 1e6:
            return f"₹{market_cap/1e6:.2f}M"
        elif market_cap >= 1e3:
            return f"₹{market_cap/1e3:.2f}K"
        else:
            return f"₹{market_cap}"
    
    def check_market_status(self) -> Dict[str, Any]:
        """
        Check if Indian markets are open
        
        Returns:
            Dictionary with market status information
        """
        now = datetime.now()
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        
        # Indian market hours: 9:15 AM to 3:30 PM IST (Monday to Friday)
        market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        is_weekend = weekday >= 5  # Saturday or Sunday
        is_market_hours = market_open_time <= now <= market_close_time
        
        if is_weekend:
            status = "Closed (Weekend)"
            next_open = "Next Monday 9:15 AM"
        elif not is_market_hours:
            if now < market_open_time:
                status = "Pre-market"
                next_open = "Today 9:15 AM"
            else:
                status = "After-market"
                next_open = "Tomorrow 9:15 AM"
        else:
            status = "Open"
            next_open = "Open until 3:30 PM"
        
        return {
            'status': status,
            'is_open': status == "Open",
            'next_open': next_open,
            'current_time': now.strftime('%Y-%m-%d %H:%M:%S'),
            'timezone': 'IST'
        }

# Create global instance
data_fetcher = DataFetcher()