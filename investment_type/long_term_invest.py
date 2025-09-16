import yfinance as yf
from groq import Groq
import os
import logging
import streamlit as st
import pandas as pd


class StockAnalyzer:
    """
    A comprehensive stock analyzer that combines quantitative data from yfinance
    with qualitative analysis from the Groq LLM, designed for Streamlit integration.
    """
    def __init__(self, ticker_symbol):
        """
        Initializes the analyzer for a given stock ticker.
        """
        self.ticker_symbol = ticker_symbol.upper()
        self.ticker = yf.Ticker(self.ticker_symbol)
        self.scores = {}
        self.max_scores = {}
        self.analysis_results = {}
        self.recommendation = ""
        self.final_score_percentage = 0

        try:
            if not os.getenv("GROQ_API_KEY"):
                logging.error("GROQ_API_KEY environment variable not set.")
                self.groq_client = None
            else:
                self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                logging.info("Groq client initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Groq client: {e}")
            self.groq_client = None

    def _get_yfinance_data(self):
        """Fetches all necessary data from yfinance."""
        logging.info(f"Fetching data for {self.ticker_symbol} from yfinance...")
        try:
            self.info = self.ticker.info
            # A more robust check for a valid ticker. A valid company should have a market cap.
            if not self.info or self.info.get('marketCap') is None or self.info.get('marketCap') == 0:
                raise ValueError(f"Invalid ticker or no data found for {self.ticker_symbol}")

            self.financials = self.ticker.financials
            self.balance_sheet = self.ticker.balance_sheet
            self.cashflow = self.ticker.cashflow
            
            if self.financials.empty or self.balance_sheet.empty:
                raise ValueError(f"Financial statements are empty for {self.ticker_symbol}. It might be a non-equity asset or delisted.")

            logging.info("Successfully fetched yfinance data.")
            return True
        except Exception as e:
            # Use a more specific error message in the UI
            st.error(f"Could not fetch complete financial data for '{self.ticker_symbol}'. The ticker might be invalid, delisted, or there could be a temporary issue with the data source. Please try another ticker.")
            logging.error(f"Could not fetch data for {self.ticker_symbol}. Error: {e}")
            return False

    def _get_financial_metric(self, data_frame, possible_keys, default=0):
        """
        Safely retrieves a financial metric from a DataFrame by checking a list of possible keys.
        """
        if data_frame.empty:
            logging.warning(f"DataFrame is empty. Cannot find keys {possible_keys}.")
            return default
            
        for key in possible_keys:
            if key in data_frame.index:
                value = data_frame.loc[key].iloc[0]
                return value if pd.notna(value) else default
        logging.warning(f"Could not find any of the keys {possible_keys} in the provided data. Returning default value {default}.")
        return default

    def _calculate_and_score_metrics(self):
        """
        Calculates derived metrics and scores them based on predefined criteria,
        adapting to the company's industry sector.
        """
        logging.info("Calculating and scoring metrics...")
        sector = self.info.get('sector', 'N/A')

        # --- Profitability Ratios ---
        net_income = self._get_financial_metric(self.financials, ['Net Income', 'NetIncome'])
        total_revenue = self._get_financial_metric(self.financials, ['Total Revenue', 'TotalRevenue'])
        total_equity = self._get_financial_metric(self.balance_sheet, ['Stockholder Equity', 'Total Stockholder Equity', 'Total Equity', 'Stockholders Equity'])

        self.max_scores['ROE'] = 10
        roe = (net_income / total_equity) * 100 if total_equity else 0
        self.analysis_results['ROE'] = f"{roe:.2f}%"
        if roe > 20: self.scores['ROE'] = 10
        elif roe > 15: self.scores['ROE'] = 8
        elif roe > 10: self.scores['ROE'] = 6
        elif roe > 5: self.scores['ROE'] = 4
        else: self.scores['ROE'] = 1

        self.max_scores['Net Margin'] = 10
        net_margin = (net_income / total_revenue) * 100 if total_revenue else 0
        self.analysis_results['Net Margin'] = f"{net_margin:.2f}%"
        if net_margin > 20: self.scores['Net Margin'] = 10
        elif net_margin > 10: self.scores['Net Margin'] = 8
        elif net_margin > 5: self.scores['Net Margin'] = 6
        else: self.scores['Net Margin'] = 3

        # --- Solvency Ratios ---
        self.max_scores['Debt/Equity'] = 10
        total_debt = self.info.get('totalDebt', 0)
        d_e_ratio = total_debt / total_equity if total_equity else float('inf')
        self.analysis_results['Debt/Equity'] = f"{d_e_ratio:.2f}"
        if d_e_ratio < 0.5: self.scores['Debt/Equity'] = 10
        elif d_e_ratio < 1.0: self.scores['Debt/Equity'] = 7
        elif d_e_ratio < 1.5: self.scores['Debt/Equity'] = 4
        else: self.scores['Debt/Equity'] = 1

        # --- Liquidity Ratios ---
        logging.info("Calculating Liquidity Ratios...")
        current_assets = self._get_financial_metric(self.balance_sheet, ['Total Current Assets', 'Current Assets'])
        current_liabilities = self._get_financial_metric(self.balance_sheet, ['Total Current Liabilities', 'Current Liabilities'])
        
        if current_assets == 0 or current_liabilities == 0:
            warning_message = (
                "**Standard liquidity ratios (Current, Quick) are not applicable.**\n\n"
                "This is common for companies in the 'Financial Services' sector, as their balance sheets do not use 'Current' classifications."
            )
            self.analysis_results['LiquidityWarning'] = warning_message
            self.analysis_results['Current Ratio'], self.analysis_results['Quick Ratio'] = "N/A", "N/A"
            self.scores['Current Ratio'], self.max_scores['Current Ratio'] = 0, 0
            self.scores['Quick Ratio'], self.max_scores['Quick Ratio'] = 0, 0
        else:
            self.max_scores['Current Ratio'] = 5
            current_ratio = current_assets / current_liabilities
            self.analysis_results['Current Ratio'] = f"{current_ratio:.2f}"
            if current_ratio > 2: self.scores['Current Ratio'] = 5
            elif current_ratio > 1.5: self.scores['Current Ratio'] = 4
            elif current_ratio > 1: self.scores['Current Ratio'] = 2
            else: self.scores['Current Ratio'] = 0

            self.max_scores['Quick Ratio'] = 5
            inventory = self._get_financial_metric(self.balance_sheet, ['Inventory'])
            quick_ratio = (current_assets - inventory) / current_liabilities
            self.analysis_results['Quick Ratio'] = f"{quick_ratio:.2f}"
            if quick_ratio > 1.5: self.scores['Quick Ratio'] = 5
            elif quick_ratio > 1: self.scores['Quick Ratio'] = 4
            elif quick_ratio > 0.5: self.scores['Quick Ratio'] = 2
            else: self.scores['Quick Ratio'] = 0

        # --- Efficiency Ratios ---
        logging.info("Calculating Efficiency Ratios...")
        self.max_scores['Asset Turnover'] = 5
        assets_series = self.balance_sheet.loc['Total Assets'] if 'Total Assets' in self.balance_sheet.index else None
        avg_total_assets = assets_series.iloc[:2].mean() if assets_series is not None and len(assets_series) > 1 else (assets_series.iloc[0] if assets_series is not None else 0)
        asset_turnover = total_revenue / avg_total_assets if avg_total_assets else 0
        self.analysis_results['Asset Turnover'] = f"{asset_turnover:.2f}"

        if sector == 'Financial Services':
            if asset_turnover > 0.08: self.scores['Asset Turnover'] = 5
            elif asset_turnover > 0.06: self.scores['Asset Turnover'] = 4
            elif asset_turnover > 0.04: self.scores['Asset Turnover'] = 3
            else: self.scores['Asset Turnover'] = 1
        else:
            if asset_turnover > 2.0: self.scores['Asset Turnover'] = 5
            elif asset_turnover > 1.2: self.scores['Asset Turnover'] = 4
            elif asset_turnover > 0.8: self.scores['Asset Turnover'] = 3
            else: self.scores['Asset Turnover'] = 1

        NON_INVENTORY_SECTORS = ['Financial Services', 'Technology']
        if sector in NON_INVENTORY_SECTORS:
            warning_message = (
                f"**Inventory Turnover is not an applicable metric.**\n\n"
                f"As a company in the '{sector}' sector, its business model is not based on selling physical inventory."
            )
            self.analysis_results['InventoryTurnoverWarning'] = warning_message
            self.analysis_results['Inventory Turnover'] = "N/A"
            self.scores['Inventory Turnover'], self.max_scores['Inventory Turnover'] = 0, 0
        else:
            self.max_scores['Inventory Turnover'] = 5
            cogs = self._get_financial_metric(self.financials, ['Cost Of Revenue', 'Cost Of Goods Sold'])
            if 'Inventory' in self.balance_sheet.index and not self.balance_sheet.loc['Inventory'].isna().all():
                inventory_series = self.balance_sheet.loc['Inventory']
                avg_inventory = inventory_series.iloc[:2].mean() if len(inventory_series) > 1 else inventory_series.iloc[0]
                if avg_inventory > 0:
                    inventory_turnover = cogs / avg_inventory
                    self.analysis_results['Inventory Turnover'] = f"{inventory_turnover:.2f}"
                    if inventory_turnover > 10: self.scores['Inventory Turnover'] = 5
                    elif inventory_turnover > 5: self.scores['Inventory Turnover'] = 4
                    elif inventory_turnover > 2: self.scores['Inventory Turnover'] = 2
                    else: self.scores['Inventory Turnover'] = 1
                else:
                    self.analysis_results['Inventory Turnover'] = "N/A"
                    self.scores['Inventory Turnover'], self.max_scores['Inventory Turnover'] = 0, 0
            else:
                self.analysis_results['Inventory Turnover'] = "N/A"
                self.scores['Inventory Turnover'], self.max_scores['Inventory Turnover'] = 0, 0
                logging.info("Calculating cash flow metrics...")

        # --- Free Cash Flow (FCF) ---
        self.max_scores['Free Cash Flow'] = 10
        operating_cash_flow = self._get_financial_metric(self.cashflow, ['Operating Cash Flow', 'Total Cash From Operating Activities'])
        # Capital Expenditure is often reported as a negative value in yfinance, so we add it to OCF.
        capital_expenditure = self._get_financial_metric(self.cashflow, ['Capital Expenditure'])
        free_cash_flow = operating_cash_flow + capital_expenditure

        # Format the FCF value for better readability (e.g., in Billions or Millions)
        if abs(free_cash_flow) >= 1e9:
            fcf_display = f"{free_cash_flow / 1e9:.2f}B"
        elif abs(free_cash_flow) >= 1e6:
            fcf_display = f"{free_cash_flow / 1e6:.2f}M"
        else:
            fcf_display = f"{free_cash_flow / 1e3:.2f}K"
        
        self.analysis_results['Free Cash Flow'] = fcf_display
        
        # A simple but crucial score: is the company generating cash?
        if free_cash_flow > 0:
            self.scores['Free Cash Flow'] = 10
        else:
            self.scores['Free Cash Flow'] = 0

        # --- Operating Cash Flow to Sales Ratio ---
        self.max_scores['OCF / Sales'] = 10
        total_revenue = self._get_financial_metric(self.financials, ['Total Revenue', 'TotalRevenue'])
        ocf_to_sales = (operating_cash_flow / total_revenue) * 100 if total_revenue else 0
        self.analysis_results['OCF / Sales'] = f"{ocf_to_sales:.2f}%"

        # Score based on how efficiently the company converts sales to cash
        if ocf_to_sales > 20:
            self.scores['OCF / Sales'] = 10
        elif ocf_to_sales > 15:
            self.scores['OCF / Sales'] = 8
        elif ocf_to_sales > 10:
            self.scores['OCF / Sales'] = 6
        elif ocf_to_sales > 5:
            self.scores['OCF / Sales'] = 4
        else:
            self.scores['OCF / Sales'] = 1
        # --- Valuation & Payout Ratios ---
        self.max_scores['P/E Ratio'] = 10
        pe_ratio = self.info.get('trailingPE')
        self.analysis_results['P/E Ratio'] = f"{pe_ratio:.2f}" if pe_ratio else "N/A"
        if pe_ratio and pe_ratio > 0:
            if pe_ratio < 15: self.scores['P/E Ratio'] = 10
            elif pe_ratio < 25: self.scores['P/E Ratio'] = 7
            elif pe_ratio < 40: self.scores['P/E Ratio'] = 4
            else: self.scores['P/E Ratio'] = 1
        else:
            self.scores['P/E Ratio'] = 0

        self.max_scores['PEG Ratio'] = 10
        peg_ratio = self.info.get('pegRatio') or self.info.get('trailingPegRatio')

        if not peg_ratio or peg_ratio <= 0 or peg_ratio > 10:
            logging.info(f"Yahoo PEG ratio is {'missing' if not peg_ratio else 'unreasonable (' + str(peg_ratio) + ')'}. Attempting manual calculation.")
            pe = self.info.get('forwardPE') or self.info.get('trailingPE')
            earnings_growth = self.info.get('earningsGrowth')

            if pe and earnings_growth and earnings_growth > 0:
                peg_ratio = pe / (earnings_growth * 100)
                logging.info(f"Manually calculated PEG: {peg_ratio:.2f}")
            else:
                peg_ratio = None
                logging.warning("Cannot manually calculate PEG ratio due to missing PE or non-positive earnings growth.")

        self.analysis_results['PEG Ratio'] = f"{peg_ratio:.2f}" if peg_ratio else "N/A"
        if peg_ratio and peg_ratio > 0:
            if peg_ratio < 1: self.scores['PEG Ratio'] = 10
            elif peg_ratio < 2: self.scores['PEG Ratio'] = 7
            else: self.scores['PEG Ratio'] = 2
        else:
            self.scores['PEG Ratio'] = 0
            
        self.max_scores['Payout Ratio'] = 5
        payout_ratio = self.info.get('payoutRatio')
        if payout_ratio is None:
            payout_ratio = 0
        self.analysis_results['Payout Ratio'] = f"{payout_ratio*100:.2f}%"
        if 0 < payout_ratio < 0.6: self.scores['Payout Ratio'] = 5
        elif payout_ratio > 0.6: self.scores['Payout Ratio'] = 2
        else:
            self.scores['Payout Ratio'] = 0
        
        logging.info("All metrics calculated and scored.")

    def _get_qualitative_analysis(self):
            """
            Generates a data-rich prompt with calculated metrics and gets a structured
            qualitative analysis from the Groq LLM.
            """
            if not self.groq_client:
                logging.warning("Groq client not available or API key not set. Skipping qualitative analysis.")
                self.analysis_results['Qualitative Analysis'] = "Groq client not initialized. Please set the GROQ_API_KEY environment variable."
                return

            logging.info("Generating data-driven qualitative analysis with Groq...")
            company_name = self.info.get('longName', self.ticker_symbol)
            sector = self.info.get('sector', 'N/A')
            industry = self.info.get('industry', 'N/A')

            # --- Create a summary of the quantitative data ---
            # This compiles our calculated metrics into a string for the LLM to analyze.
            quantitative_summary = f"""
            - Sector: {sector}, Industry: {industry}
            - Return on Equity (ROE): {self.analysis_results.get('ROE', 'N/A')}
            - Net Profit Margin: {self.analysis_results.get('Net Margin', 'N/A')}
            - Debt/Equity Ratio: {self.analysis_results.get('Debt/Equity', 'N/A')}
            - Current Ratio: {self.analysis_results.get('Current Ratio', 'N/A')}
            - Quick Ratio: {self.analysis_results.get('Quick Ratio', 'N/A')}
            - Asset Turnover: {self.analysis_results.get('Asset Turnover', 'N/A')}
            - Inventory Turnover: {self.analysis_results.get('Inventory Turnover', 'N/A')}
            - Free Cash Flow: {self.analysis_results.get('Free Cash Flow', 'N/A')}
            - OCF / Sales Ratio: {self.analysis_results.get('OCF / Sales', 'N/A')}
            - P/E Ratio: {self.analysis_results.get('P/E Ratio', 'N/A')}
            - PEG Ratio: {self.analysis_results.get('PEG Ratio', 'N/A')}
            - Payout Ratio: {self.analysis_results.get('Payout Ratio', 'N/A')}
            - News Sentiment: {self.analysis_results.get('Sentiment Label', 'N/A')}
            """

            # --- Create the new, more detailed prompt ---
            prompt = f"""
            Act as an expert financial analyst reviewing **{company_name} ({self.ticker_symbol})** for a long-term investor.

            Here is a summary of its key quantitative metrics:
            {quantitative_summary}
                    **Important Note:** The financial data is from an automated source (yfinance) and may occasionally contain errors or missing values. If a specific metric seems unusually high or low compared to industry norms, please mention this as a point of caution in your analysis.
            always 
            Based on this data and your general knowledge, provide a concise, expert-level qualitative analysis. Structure your response in the following format, using Markdown for clear headings:

            **1. Overall Summary:**
            Provide a brief, high-level overview of the company's financial health and investment profile based on the data.

            **2. Key Strengths:**
            Based on the provided metrics, identify and explain the company's strongest financial attributes. (e.g., "Strong profitability as shown by a high ROE of X%").

            **3. Key Weaknesses & Risks:**
            Based on the provided metrics, identify and explain the most significant weaknesses or risks. (e.g., "High debt level indicated by a Debt/Equity ratio of Y").

            **4. Final Recommendation:**
            Conclude with a final investment recommendation (e.g., Buy, Hold, Avoid) and a brief justification for your reasoning.
            """
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                )
                self.analysis_results['Qualitative Analysis'] = chat_completion.choices[0].message.content
                logging.info("Successfully generated data-driven qualitative analysis.")
            except Exception as e:
                logging.error(f"Failed to get qualitative analysis from Groq: {e}")
                self.analysis_results['Qualitative Analysis'] = f"Error fetching analysis from Groq: {e}"
                
    def _generate_final_verdict(self):
        total_score = sum(self.scores.values())
        max_score = sum(self.max_scores.values())
        self.final_score_percentage = (total_score / max_score) * 100 if max_score > 0 else 0

        if self.final_score_percentage >= 80:
            self.recommendation = "EXCELLENT candidate for long-term investment. Strong fundamentals and a positive outlook."
        elif self.final_score_percentage >= 60:
            self.recommendation = "GOOD candidate for long-term investment. Solid fundamentals, but some areas warrant closer inspection."
        elif self.final_score_percentage >= 40:
            self.recommendation = "NEUTRAL. The company has mixed fundamentals. Further research is essential before investing."
        else:
            self.recommendation = "CAUTION ADVISED. Significant weaknesses in fundamentals. May not be suitable for a conservative long-term investor."

    def run_analysis(self):
        if self._get_yfinance_data():
            self._calculate_and_score_metrics()
            # <-- ADD THIS LINE
            self._get_qualitative_analysis()
            self._generate_final_verdict()
            return {
                "info": self.info,
                "quantitative": self.analysis_results,
                "scores": self.scores,
                "max_scores": self.max_scores,
                "qualitative": self.analysis_results.get('Qualitative Analysis'),
                "final_score": self.final_score_percentage,
                "recommendation": self.recommendation
            }
        return None