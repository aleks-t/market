import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import anthropic
import os
import warnings
warnings.filterwarnings('ignore')

# Check for required environment variables
if not os.getenv('ANTHROPIC_API_KEY'):
    st.warning("⚠️ ANTHROPIC_API_KEY environment variable not set. AI analysis will be disabled.")

# Page config
st.set_page_config(
    page_title="Market Scout",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
    }
    .stProgress>div>div>div {
        background-color: #00cc00;
    }
    </style>
    """, unsafe_allow_html=True)

class MarketScout:
    def __init__(self):
        self.results = []

    def check_condition(self, data, condition, date_idx):
        """Check if trigger condition is met at given date index"""
        try:
            if date_idx < condition.get('lookback_days', 1):
                return False

            condition_type = condition['type']

            if condition_type == 'price_change':
                lookback = condition.get('lookback_days', 1)
                threshold = float(condition['threshold_pct'])
                direction = condition['direction']

                close_prices = data['Close']
                start_price = float(close_prices.iloc[date_idx - lookback])
                end_price = float(close_prices.iloc[date_idx])

                pct_change = ((end_price - start_price) / start_price) * 100

                if direction == 'up':
                    return pct_change >= threshold
                else:
                    return pct_change <= -threshold

            elif condition_type == 'above_ma':
                ma_period = condition['ma_period']
                if date_idx < ma_period:
                    return False

                close_prices = data['Close']
                ma_data = close_prices.iloc[date_idx-ma_period:date_idx]
                ma = float(ma_data.mean())
                current_price = float(close_prices.iloc[date_idx])

                return current_price > ma

            elif condition_type == 'volume_spike':
                lookback = condition['lookback_days']
                multiplier = condition['multiplier']
                if date_idx < lookback:
                    return False

                volume_data = data['Volume']
                avg_volume = float(volume_data.iloc[date_idx-lookback:date_idx].mean())
                current_volume = float(volume_data.iloc[date_idx])

                return current_volume > (avg_volume * multiplier)

            return False

        except Exception:
            return False

    def calculate_forward_returns(self, target_data, signal_date, forward_periods):
        """Calculate returns at multiple forward periods from signal date - FIXED VERSION"""
        try:
            signal_idx = target_data.index.get_loc(signal_date)
            signal_price = float(target_data.iloc[signal_idx]['Close'])

            returns = {}

            for period in forward_periods:
                target_idx = signal_idx + period

                if target_idx < len(target_data):
                    future_price = float(target_data.iloc[target_idx]['Close'])
                    forward_return = ((future_price - signal_price) / signal_price) * 100
                    returns[f'{period}d'] = forward_return
                else:
                    returns[f'{period}d'] = None

            return returns

        except Exception:
            return {}

    def run_experiment(self, experiment, silent=False):
        """Run experiment with proper forward-looking analysis"""
        try:
            if not silent:
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("📈 Fetching trigger symbol data...")
                progress_bar.progress(25)

            trigger_data, trigger_error = get_price_data(
                experiment['trigger_symbol'],
                experiment['start_date'],
                experiment['end_date']
            )

            if trigger_error:
                if not silent:
                    st.error(f"❌ Trigger data error: {trigger_error}")
                return self._empty_result(experiment['name'], trigger_error)

            if not silent:
                status_text.text("🎯 Fetching target symbol data...")
                progress_bar.progress(50)

            max_forward_days = max(experiment['forward_periods'])
            extended_end = pd.to_datetime(experiment['end_date']) + timedelta(days=max_forward_days + 50)

            target_data, target_error = get_price_data(
                experiment['target_symbol'],
                experiment['start_date'],
                extended_end.strftime('%Y-%m-%d')
            )

            if target_error:
                if not silent:
                    st.error(f"❌ Target data error: {target_error}")
                return self._empty_result(experiment['name'], target_error)

            if not silent:
                status_text.text("🔍 Scanning for signal patterns...")
                progress_bar.progress(75)

            signals = []

            for i in range(len(trigger_data)):
                try:
                    condition_met = self.check_condition(
                        trigger_data,
                        experiment['trigger_condition'],
                        i
                    )

                    if condition_met:
                        signal_date = trigger_data.index[i]

                        if signal_date in target_data.index:
                            forward_returns = self.calculate_forward_returns(
                                target_data,
                                signal_date,
                                experiment['forward_periods']
                            )

                            if any(v is not None for v in forward_returns.values()):
                                trigger_price = float(trigger_data.iloc[i]['Close'])
                                target_price = float(target_data.loc[signal_date]['Close'])

                                signals.append({
                                    'date': signal_date.strftime('%Y-%m-%d'),
                                    'trigger_price': trigger_price,
                                    'target_price': target_price,
                                    **forward_returns
                                })

                except Exception:
                    continue

            if not silent:
                status_text.text("📊 Calculating performance metrics...")
                progress_bar.progress(100)

            results = {
                'experiment': experiment['name'],
                'trigger_symbol': experiment['trigger_symbol'],
                'target_symbol': experiment['target_symbol'],
                'signals': len(signals),
                'signal_details': signals,
                'forward_analysis': {}
            }

            for period in experiment['forward_periods']:
                period_key = f'{period}d'
                valid_returns = [
                    s[period_key] for s in signals
                    if period_key in s and s[period_key] is not None
                ]

                if valid_returns:
                    arr = np.array(valid_returns)
                    results['forward_analysis'][period_key] = {
                        'valid_signals': len(valid_returns),
                        'avg_return':    float(np.mean(arr)),
                        'win_rate':      float((arr > 0).mean() * 100),
                        'best_signal':   float(np.max(arr)),
                        'worst_signal':  float(np.min(arr)),
                        'std_dev':       float(np.std(arr)),
                        'total_return':  float(np.sum(arr))
                    }
                else:
                    results['forward_analysis'][period_key] = {
                        'valid_signals': 0, 'avg_return': 0, 'win_rate': 0,
                        'best_signal': 0, 'worst_signal': 0, 'std_dev': 0, 'total_return': 0
                    }

            if not silent:
                import time
                status_text.text("✅ Analysis complete!")
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()

            return results

        except Exception as e:
            if not silent:
                st.error(f"❌ Experiment failed: {str(e)}")
            return self._empty_result(experiment['name'], str(e))

    def _empty_result(self, name, error):
        """Return empty result structure for failed experiments"""
        return {
            'experiment': name,
            'signals': 0,
            'forward_analysis': {},
            'signal_details': [],
            'error': error
        }

# Utility Functions
SYMBOL_MAP = {
    # ── Crypto ──────────────────────────────────────────────────────────────
    'BTC': 'BTC-USD', 'BITCOIN': 'BTC-USD',
    'ETH': 'ETH-USD', 'ETHEREUM': 'ETH-USD',
    'SOL': 'SOL-USD', 'BNB': 'BNB-USD', 'XRP': 'XRP-USD',

    # ── US Macro ─────────────────────────────────────────────────────────────
    'DXY': 'DX-Y.NYB', 'DOLLAR': 'DX-Y.NYB',
    'VIX': '^VIX', 'FEAR': '^VIX',
    'TNX': '^TNX', '10Y': '^TNX', '10YR': '^TNX',
    'TYX': '^TYX', '30Y': '^TYX',
    'FVX': '^FVX', '5Y': '^FVX',

    # ── Precious Metals ───────────────────────────────────────────────────────
    'GOLD': 'GC=F', 'XAU': 'GC=F',
    'SILVER': 'SI=F', 'XAG': 'SI=F',
    'PLATINUM': 'PL=F', 'XPT': 'PL=F',
    'PALLADIUM': 'PA=F', 'XPD': 'PA=F',

    # ── Industrial Metals ─────────────────────────────────────────────────────
    'COPPER': 'HG=F', 'CU': 'HG=F',
    'ALUMINUM': 'ALI=F', 'ALUMINIUM': 'ALI=F', 'AL': 'ALI=F',
    'STEEL': 'SLX',        # Steel ETF — no clean futures via yfinance
    'IRON': 'SLX',         # Closest proxy

    # ── Energy ────────────────────────────────────────────────────────────────
    'OIL': 'CL=F', 'CRUDE': 'CL=F', 'WTI': 'CL=F',
    'BRENT': 'BZ=F',
    'GAS': 'NG=F', 'NATGAS': 'NG=F', 'NATURALGAS': 'NG=F',
    'GASOLINE': 'RB=F',
    'HEATING': 'HO=F',

    # ── Agricultural ──────────────────────────────────────────────────────────
    'CORN': 'ZC=F',
    'WHEAT': 'ZW=F',
    'SOYBEANS': 'ZS=F', 'SOY': 'ZS=F',
    'SOYOIL': 'ZL=F',
    'COFFEE': 'KC=F',
    'SUGAR': 'SB=F',
    'COTTON': 'CT=F',
    'COCOA': 'CC=F',
    'LUMBER': 'LBS=F',

    # ── US Indices ────────────────────────────────────────────────────────────
    'SPX': '^GSPC', 'SP500': '^GSPC', 'S&P': '^GSPC',
    'NDX': '^NDX', 'NASDAQ': '^IXIC',
    'DOW': '^DJI', 'DJIA': '^DJI',
    'RUT': '^RUT', 'RUSSELL': '^RUT',

    # ── International Indices ─────────────────────────────────────────────────
    'FTSE': '^FTSE', 'UK': '^FTSE',
    'DAX': '^GDAXI', 'GERMANY': '^GDAXI',
    'CAC': '^FCHI', 'FRANCE': '^FCHI',
    'NIKKEI': '^N225', 'JAPAN': '^N225',
    'HANGSENG': '^HSI', 'HSI': '^HSI', 'HK': '^HSI',
    'SHANGHAI': '000001.SS', 'SSE': '000001.SS',
    'ASX': '^AXJO', 'AUSTRALIA': '^AXJO',
    'SENSEX': '^BSESN', 'INDIA': '^BSESN',
    'NIFTY': '^NSEI',
    'BOVESPA': '^BVSP', 'BRAZIL': '^BVSP',
    'TSX': '^GSPTSE', 'CANADA': '^GSPTSE',
    'KOSPI': '^KS11', 'KOREA': '^KS11',
    'TAIEX': '^TWII', 'TAIWAN': '^TWII',

    # ── International ETFs (easier than index tickers) ────────────────────────
    'CHINA': 'FXI',
    'EUROPE': 'VGK',
    'EM': 'EEM', 'EMERGING': 'EEM',
    'INTL': 'EFA', 'INTERNATIONAL': 'EFA',

    # ── Forex ─────────────────────────────────────────────────────────────────
    'EURUSD': 'EURUSD=X', 'EUR': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X', 'GBP': 'GBPUSD=X', 'POUND': 'GBPUSD=X',
    'USDJPY': 'USDJPY=X', 'JPY': 'USDJPY=X', 'YEN': 'USDJPY=X',
    'USDCAD': 'USDCAD=X', 'CAD': 'USDCAD=X',
    'AUDUSD': 'AUDUSD=X', 'AUD': 'AUDUSD=X',
    'USDCHF': 'USDCHF=X', 'CHF': 'USDCHF=X',
    'USDCNY': 'USDCNY=X', 'CNY': 'USDCNY=X',
}

# Asset type classification — used to pick the right stress-test years
ASSET_TYPES = {
    'commodity_energy':    {'CL=F','BZ=F','NG=F','RB=F','HO=F'},
    'commodity_metal':     {'GC=F','SI=F','HG=F','PL=F','PA=F','ALI=F','SLX'},
    'commodity_agri':      {'ZC=F','ZW=F','ZS=F','ZL=F','KC=F','SB=F','CT=F','CC=F','LBS=F'},
    'crypto':              {'BTC-USD','ETH-USD','SOL-USD','BNB-USD','XRP-USD'},
    'forex':               {'DX-Y.NYB','EURUSD=X','GBPUSD=X','USDJPY=X','USDCAD=X','AUDUSD=X','USDCHF=X','USDCNY=X'},
    'intl_developed':      {'^FTSE','^GDAXI','^FCHI','^N225','^AXJO','^GSPTSE','^TWII','^KS11','EFA','VGK','EWJ','EWG','EWU','EWA'},
    'intl_emerging':       {'^HSI','000001.SS','^BSESN','^NSEI','^BVSP','^KS11','EEM','FXI','EWZ','INDA'},
}

# Stress-test year ranges by asset type
# 2022 is a bad stress test for commodities — energy/metals surged that year
STRESS_TEST_YEARS = {
    'commodity_energy':  [
        ('2023  (oil/energy selloff)',      '2023-01-01', '2023-12-31'),
        ('2020  (COVID crash)',             '2020-01-01', '2020-12-31'),
        ('2018  (energy bear)',             '2018-01-01', '2018-12-31'),
        ('Last 12 months',                 None, None),
    ],
    'commodity_metal':   [
        ('2022  (mixed — metals volatile)', '2022-01-01', '2022-12-31'),
        ('2023  (metals selloff)',          '2023-01-01', '2023-12-31'),
        ('2018  (metals bear)',             '2018-01-01', '2018-12-31'),
        ('Last 12 months',                 None, None),
    ],
    'commodity_agri':    [
        ('2023  (agri retreat)',            '2023-01-01', '2023-12-31'),
        ('2019  (agri bear)',               '2019-01-01', '2019-12-31'),
        ('Last 12 months',                 None, None),
    ],
    'crypto':            [
        ('2022  (crypto winter, -65%)',     '2022-01-01', '2022-12-31'),
        ('2018  (crypto crash, -80%)',      '2018-01-01', '2018-12-31'),
        ('2023  (recovery)',                '2023-01-01', '2023-12-31'),
        ('Last 12 months',                 None, None),
    ],
    'forex':             [
        ('2022  (dollar surge)',            '2022-01-01', '2022-12-31'),
        ('2020  (COVID volatility)',        '2020-01-01', '2020-12-31'),
        ('Last 12 months',                 None, None),
    ],
    'intl_developed':    [
        ('2022  (global selloff)',          '2022-01-01', '2022-12-31'),
        ('2018  (intl bear)',               '2018-01-01', '2018-12-31'),
        ('2023  (recovery)',                '2023-01-01', '2023-12-31'),
        ('Last 12 months',                 None, None),
    ],
    'intl_emerging':     [
        ('2022  (EM bear)',                 '2022-01-01', '2022-12-31'),
        ('2018  (EM crisis)',               '2018-01-01', '2018-12-31'),
        ('2023',                            '2023-01-01', '2023-12-31'),
        ('Last 12 months',                 None, None),
    ],
    'default':           [
        ('2022  (bear year — S&P -18%)',    '2022-01-01', '2022-12-31'),
        ('2023  (recovery)',                '2023-01-01', '2023-12-31'),
        ('2024',                            '2024-01-01', '2024-12-31'),
        ('Last 12 months',                 None, None),
    ],
}

def classify_asset(ticker):
    """Return the asset category for a given yfinance ticker"""
    for asset_type, tickers in ASSET_TYPES.items():
        if ticker in tickers:
            return asset_type
    if ticker.endswith('=F'):
        return 'commodity_metal'
    if ticker.endswith('-USD'):
        return 'crypto'
    if ticker.endswith('=X'):
        return 'forex'
    if ticker.startswith('^'):
        return 'intl_developed'
    return 'default'

def format_symbol(symbol):
    """Resolve common shorthand names to Yahoo Finance tickers"""
    symbol = symbol.upper().strip()
    return SYMBOL_MAP.get(symbol, symbol)

def get_price_data(symbol, start_date, end_date):
    """Fetch price data with robust error handling"""
    try:
        formatted_symbol = format_symbol(symbol)
        
        for attempt in range(3):
            try:
                data = yf.download(
                    formatted_symbol,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                    keepna=False,
                    threads=False
                )
                
                if not data.empty:
                    break
                    
            except Exception:
                if attempt == 2:
                    return None, f"Failed to fetch data for {symbol} after 3 attempts"
                continue
        
        if data.empty:
            return None, f"No data available for {symbol} ({formatted_symbol})"
        
        # Handle multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Handle columns with same names
        if len(data.columns) >= 4 and len(set(data.columns)) == 1:
            expected_names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            actual_names = expected_names[:len(data.columns)]
            data.columns = actual_names
        
        # Find Close column
        close_column = None
        for col_name in ['Close', 'Adj Close', 'close', 'adj_close']:
            if col_name in data.columns:
                close_column = col_name
                break
        
        if close_column is None:
            return None, f"No Close price column found for {symbol}"
        
        # Rename to standard 'Close'
        if close_column != 'Close':
            data = data.rename(columns={close_column: 'Close'})
        
        # Ensure Volume column
        if 'Volume' not in data.columns:
            data['Volume'] = 1000000
        
        # Remove NaN values
        data = data.dropna(subset=['Close'])
        
        if len(data) < 10:
            return None, f"Insufficient data for {symbol} (only {len(data)} rows)"
        
        return data, None
        
    except Exception as e:
        return None, f"Error fetching {symbol}: {str(e)}"

@st.cache_data(show_spinner=False)
def get_ai_analysis(experiment_name, trigger_symbol, target_symbol, results):
    """Generate enhanced AI analysis with environment variable API key"""
    try:
        api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not api_key:
            return "❌ AI analysis disabled: ANTHROPIC_API_KEY environment variable not set."

        client = anthropic.Anthropic(api_key=api_key)

        # Get the primary analysis period
        forward_analysis = results.get('forward_analysis', {})
        primary_period = None
        primary_stats = None

        for period in ['90d', '60d', '30d', '15d', '7d']:
            if period in forward_analysis and forward_analysis[period]['valid_signals'] > 0:
                primary_period = period
                primary_stats = forward_analysis[period]
                break

        if not primary_stats:
            return "❌ No valid forward returns found for analysis."

        prompt = f"""Analyze this market pattern like Adam Robinson would - focus on practical trading insights.

PATTERN: {experiment_name}
TRIGGER: {trigger_symbol} → TARGET: {target_symbol}

KEY RESULTS ({primary_period} forward returns):
• Total Signals: {primary_stats['valid_signals']}
• Average Return: {primary_stats['avg_return']:.2f}%
• Win Rate: {primary_stats['win_rate']:.1f}%
• Best Trade: +{primary_stats['best_signal']:.2f}%
• Worst Trade: {primary_stats['worst_signal']:.2f}%
• Standard Deviation: {primary_stats['std_dev']:.2f}%
• Total Cumulative Return: {primary_stats['total_return']:.2f}%

MULTI-TIMEFRAME ANALYSIS:
{chr(10).join([f"• {period}: {stats['avg_return']:.2f}% avg, {stats['win_rate']:.1f}% win rate ({stats['valid_signals']} signals)" 
               for period, stats in forward_analysis.items() if stats['valid_signals'] > 0])}

Start with a practical summary in this exact format:

💡 **PRACTICAL TRADING SUMMARY**

"When {trigger_symbol} rallies [trigger condition], buy {target_symbol} and hold for {primary_period.replace('d', ' days')}. You'll win {primary_stats['win_rate']:.0f}% of the time (about {int(primary_stats['win_rate'])//10} out of 10 trades) with an average {primary_stats['avg_return']:.1f}% return, but could lose up to {abs(primary_stats['worst_signal']):.1f}% on bad trades."

Then provide detailed analysis in these sections:

📊 **PATTERN QUALITY ASSESSMENT**
- Is this win rate statistically significant?
- How does the risk/reward compare to buy-and-hold?
- What's the practical value of this edge?

💰 **TRADING STRATEGY**
- Exact entry rules based on the trigger
- Position sizing recommendations
- Risk management (stop loss levels)
- Optimal holding period based on timeframes

⚠️ **RISK ANALYSIS**
- Maximum drawdown potential
- When this pattern might fail
- Market conditions to avoid
- Portfolio allocation suggestions

🔄 **OPTIMIZATION IDEAS**
- How to improve the win rate
- Additional filters to consider
- Related patterns to stack
- Market regime considerations

🎯 **BOTTOM LINE**
- Is this pattern worth trading?
- Expected annual returns if followed
- Real-world implementation challenges

Be specific with numbers and actionable advice. Compare to realistic market benchmarks."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )

        return message.content[0].text

    except Exception as e:
        return f"❌ **AI Analysis Error**: {str(e)}\n\nThe API key may have expired or there might be a rate limit issue."

# Session State Initialization
if 'experiments' not in st.session_state:
    st.session_state.experiments = []

if 'scout' not in st.session_state:
    st.session_state.scout = MarketScout()

def main():
    """Main application"""
    # Header
    st.title("🔍 Market Scout")
    
    # Add session reset button for debugging
    if st.button("🔄 Reset Session (if experiencing issues)"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Session reset! Please refresh the page.")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("Create Experiment")

        # Quick presets
        st.write("**📚 US Equity Patterns**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("GLD → GDX", use_container_width=True):    load_preset('gld-gdx')
            if st.button("VIX → SPY", use_container_width=True):    load_preset('vix-spy')
            if st.button("BTC → QQQ", use_container_width=True):    load_preset('btc-nasdaq')
        with col2:
            if st.button("DXY → Gold", use_container_width=True):   load_preset('dxy-gold')
            if st.button("IWM → SPY", use_container_width=True):    load_preset('iwm-spy')
            if st.button("Oil → XLE", use_container_width=True):    load_preset('oil-energy')

        st.write("**⛏️ Commodities**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Gold → Silver", use_container_width=True):   load_preset('gold-silver')
            if st.button("Oil → Copper", use_container_width=True):    load_preset('oil-copper')
        with col2:
            if st.button("Copper → SLX", use_container_width=True):    load_preset('copper-steel')
            if st.button("Wheat → Corn", use_container_width=True):    load_preset('wheat-corn')

        st.write("**🌍 International**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("SPY → Nikkei", use_container_width=True):    load_preset('spy-nikkei')
            if st.button("DXY → EEM", use_container_width=True):       load_preset('dxy-eem')
        with col2:
            if st.button("SPY → DAX", use_container_width=True):       load_preset('spy-dax')
            if st.button("BTC → FXI", use_container_width=True):       load_preset('btc-china')

        st.divider()

        # Experiment form
        trigger_symbol = st.text_input(
            "Trigger Symbol",
            value=st.session_state.get('trigger_symbol', ''),
            help="Type any ticker — or use shorthand: GOLD, COPPER, WHEAT, NIKKEI, DAX, BTC, OIL, etc."
        )

        target_symbol = st.text_input(
            "Target Symbol",
            value=st.session_state.get('target_symbol', ''),
            help="🎯 The asset whose returns we measure (e.g., GDX, SPY, QQQ)"
        )

        # Trigger condition
        st.write("**🎯 Trigger Condition**")
        condition_type = st.selectbox(
            "Condition Type",
            ["Price Change", "Above Moving Average", "Volume Spike"],
            help="Price Change: When X moves Y%, what happens to Z? (Most popular)"
        )

        if condition_type == "Price Change":
            col1, col2 = st.columns(2)
            with col1:
                direction = st.selectbox("Direction", ["Up", "Down"])
            with col2:
                threshold = st.number_input("Threshold %", value=2.0, min_value=0.1, step=0.1)
            lookback_days = st.number_input("Lookback Days", value=1, min_value=1, max_value=10)

        elif condition_type == "Above Moving Average":
            ma_period = st.number_input("MA Period", value=50, min_value=5, max_value=200)

        elif condition_type == "Volume Spike":
            col1, col2 = st.columns(2)
            with col1:
                vol_lookback = st.number_input("Lookback Days", value=20, min_value=5)
            with col2:
                vol_multiplier = st.number_input("Volume Multiplier", value=2.0, min_value=1.1, step=0.1)

        # Analysis period
        st.write("**📅 Analysis Period**")
        col1, col2 = st.columns(2)
        with col1:
            days_back = st.number_input("Days Back", value=365, min_value=90, max_value=1825)
        with col2:
            end_date = st.date_input("End Date", value=datetime.now().date())

        start_date = end_date - timedelta(days=days_back)

        # Forward periods
        st.write("**📈 Forward Analysis Periods**")
        col1, col2 = st.columns(2)
        with col1:
            analyze_7d = st.checkbox("7 days", value=True)
            analyze_30d = st.checkbox("30 days", value=True)
            analyze_90d = st.checkbox("90 days", value=False)

        with col2:
            analyze_15d = st.checkbox("15 days", value=False)
            analyze_60d = st.checkbox("60 days", value=False)
            analyze_180d = st.checkbox("180 days", value=False)

        # Build forward periods list
        forward_periods = []
        if analyze_7d: forward_periods.append(7)
        if analyze_15d: forward_periods.append(15)
        if analyze_30d: forward_periods.append(30)
        if analyze_60d: forward_periods.append(60)
        if analyze_90d: forward_periods.append(90)
        if analyze_180d: forward_periods.append(180)

        # AI analysis option
        enable_ai = st.checkbox("🤖 Enable AI Analysis", value=True)

        # Add experiment
        if st.button("Add Experiment", use_container_width=True, type="primary"):
            add_experiment(trigger_symbol, target_symbol, condition_type, 
                         direction if condition_type == "Price Change" else None,
                         threshold if condition_type == "Price Change" else None,
                         lookback_days if condition_type == "Price Change" else None,
                         ma_period if condition_type == "Above Moving Average" else None,
                         vol_lookback if condition_type == "Volume Spike" else None,
                         vol_multiplier if condition_type == "Volume Spike" else None,
                         forward_periods, start_date, end_date, enable_ai)

    # Main content
    if st.session_state.experiments:
        st.header("🧪 Experiments")

        # Show experiments
        for i, exp in enumerate(st.session_state.experiments):
            with st.expander(f"📊 {exp['name']}", expanded=False):
                col1, col2, col3 = st.columns([3, 3, 1])
                with col1:
                    st.write(f"**Trigger:** {exp['trigger_symbol']}")
                    st.write(f"**Target:** {exp['target_symbol']}")
                with col2:
                    st.write(f"**Period:** {exp['start_date']} to {exp['end_date']}")
                    ai_status = "🤖 AI Enabled" if exp.get('enable_ai', True) else "📊 AI Disabled"
                    st.write(f"**Analysis:** {ai_status}")
                with col3:
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.experiments.pop(i)
                        st.rerun()

        st.divider()

        # Run button
        if st.button("🚀 Run All Experiments", use_container_width=True, type="primary"):
            run_all_experiments()

    else:
        st.info("👆 Create experiments using the sidebar, or try a preset!")

        # Pattern Discovery Guide
        with st.expander("🧠 Adam Robinson's Pattern Discovery Guide", expanded=False):
            st.markdown("""
            ## 🎯 **How to Think Like Adam Robinson**
            
            ### **Rule #1: Start with Economic Logic**
            **❌ Don't do**: Test random symbols hoping to find patterns  
            **✅ Do**: Ask "What SHOULD be connected?" then test if it's true
            
            ### **Rule #2: Follow the Money Flows**
            **Examples:**
            - **Clean Energy Hype**: TSLA ↗️ → Solar stocks, uranium, copper
            - **AI Boom**: NVDA ↗️ → Cloud storage, data centers, semiconductors  
            - **Inflation Fear**: Gold ↗️ → TIPS, mining stocks, real estate
            
            ### **Rule #3: Look for Leading Indicators**
            **Market Leaders:**
            - **VIX** (fear) → Everything else
            - **Dollar (DXY)** → International markets
            - **10-Year Treasury (TNX)** → All interest-sensitive sectors
            - **Oil** → Transportation, inflation expectations
            
            ### **Quick Reference: Top 10 Tests to Start With**
            
            **Market Leaders & Fear:**
            1. **VIX → SPY** - Fear spikes predict market bounces
            2. **DXY → GLD** - Strong dollar crushes gold
            3. **TNX → XLU** - Rising rates kill utilities
            4. **IWM → SPY** - Small caps lead market sentiment
            
            **Sector Leadership:**
            5. **XLF → SPY** - Banks predict market health  
            6. **TSM → SMH** - Chip leader predicts sector
            7. **TSLA → ICLN** - EV leader predicts clean energy
            8. **USO → XLE** - Oil commodity leads energy stocks
            
            **Risk Appetite:**
            9. **BTC → QQQ** - Crypto signals tech risk appetite
            10. **EEM → SPY** - Emerging markets show global risk
            """)

        with st.expander("💡 How This Works"):
            st.write("""
            1. **🔍 Finds Trigger Signals**: Scans for when your trigger symbol meets the condition
            2. **📈 Measures Forward Returns**: For each signal, calculates target symbol returns
            3. **📊 Statistical Analysis**: Shows win rates, average returns, and risk metrics
            4. **🔍 Auto Reality Check**: If results look suspicious, re-runs on stress-test years automatically
            5. **🤖 AI Insights**: Provides trading strategy recommendations

            **Example**: If GLD spikes 2% on Monday, what does GDX return by Friday (7d), next month (30d), etc.
            """)

        with st.expander("📖 Ticker Reference — What can I type?"):
            st.markdown("""
**You can type shorthand names** — the app converts them automatically.

| What you type | What it means |
|---|---|
| **US Equities** | Just use the ticker: `SPY`, `QQQ`, `AAPL`, `GLD`, `GDX`, `XLE`, `IWM` |
| `GOLD` or `XAU` | Gold futures (GC=F) |
| `SILVER` or `XAG` | Silver futures (SI=F) |
| `COPPER` or `CU` | Copper futures (HG=F) |
| `STEEL` | Steel ETF (SLX) — no direct futures available |
| `OIL`, `CRUDE`, `WTI` | Crude oil futures (CL=F) |
| `BRENT` | Brent crude (BZ=F) |
| `GAS`, `NATGAS` | Natural gas (NG=F) |
| `WHEAT` | Wheat futures (ZW=F) |
| `CORN` | Corn futures (ZC=F) |
| `SOYBEANS`, `SOY` | Soybean futures (ZS=F) |
| `COFFEE` | Coffee futures (KC=F) |
| **Indices** | |
| `VIX` or `FEAR` | Volatility index (^VIX) |
| `DXY` or `DOLLAR` | US Dollar index |
| `NIKKEI` or `JAPAN` | Nikkei 225 (^N225) |
| `DAX` or `GERMANY` | German DAX (^GDAXI) |
| `FTSE` or `UK` | UK FTSE 100 (^FTSE) |
| `HANGSENG` or `HK` | Hang Seng (^HSI) |
| `SENSEX` or `INDIA` | BSE Sensex (^BSESN) |
| `BOVESPA` or `BRAZIL` | Brazilian index (^BVSP) |
| **International ETFs** | |
| `CHINA` | China ETF (FXI) |
| `EMERGING` or `EM` | Emerging markets (EEM) |
| `EUROPE` | European stocks (VGK) |
| **Crypto** | |
| `BTC` or `BITCOIN` | Bitcoin (BTC-USD) |
| `ETH` or `ETHEREUM` | Ethereum (ETH-USD) |
| **Forex** | |
| `EUR`, `EURUSD` | Euro/Dollar (EURUSD=X) |
| `GBP`, `POUND` | British pound (GBPUSD=X) |
| `JPY`, `YEN` | Japanese yen (USDJPY=X) |

**For international individual stocks**, use Yahoo Finance format: `HSBA.L` (HSBC London), `7203.T` (Toyota Tokyo), `ASML.AS` (ASML Amsterdam).
            """)

def load_preset(preset_name):
    """Load preset experiment configurations"""
    presets = {
        # US Equity
        'gld-gdx':      {'trigger': 'GLD',      'target': 'GDX'},
        'vix-spy':      {'trigger': '^VIX',     'target': 'SPY'},
        'btc-nasdaq':   {'trigger': 'BTC-USD',  'target': 'QQQ'},
        'dxy-gold':     {'trigger': 'DX-Y.NYB', 'target': 'GLD'},
        'iwm-spy':      {'trigger': 'IWM',      'target': 'SPY'},
        'oil-energy':   {'trigger': 'USO',      'target': 'XLE'},
        # Commodities
        'gold-silver':  {'trigger': 'GC=F',     'target': 'SI=F'},
        'oil-copper':   {'trigger': 'CL=F',     'target': 'HG=F'},
        'copper-steel': {'trigger': 'HG=F',     'target': 'SLX'},
        'wheat-corn':   {'trigger': 'ZW=F',     'target': 'ZC=F'},
        # International
        'spy-nikkei':   {'trigger': 'SPY',      'target': '^N225'},
        'dxy-eem':      {'trigger': 'DX-Y.NYB', 'target': 'EEM'},
        'spy-dax':      {'trigger': 'SPY',      'target': '^GDAXI'},
        'btc-china':    {'trigger': 'BTC-USD',  'target': 'FXI'},
    }

    if preset_name in presets:
        p = presets[preset_name]
        st.session_state.trigger_symbol = p['trigger']
        st.session_state.target_symbol = p['target']
        st.rerun()

def add_experiment(trigger, target, condition_type, direction, threshold, lookback, 
                  ma_period, vol_lookback, vol_multiplier, forward_periods, 
                  start_date, end_date, enable_ai=True):
    """Add experiment to session state"""
    if not all([trigger, target]) or not forward_periods:
        st.error("Please fill in trigger symbol, target symbol, and select at least one forward period")
        return

    # Build condition
    if condition_type == "Price Change":
        condition = {
            'type': 'price_change',
            'direction': direction.lower(),
            'threshold_pct': threshold,
            'lookback_days': lookback
        }
        condition_desc = f"{direction} {threshold}% in {lookback}d"
    elif condition_type == "Above Moving Average":
        condition = {
            'type': 'above_ma',
            'ma_period': ma_period
        }
        condition_desc = f"Above {ma_period}MA"
    elif condition_type == "Volume Spike":
        condition = {
            'type': 'volume_spike',
            'lookback_days': vol_lookback,
            'multiplier': vol_multiplier
        }
        condition_desc = f"Vol {vol_multiplier}x spike"

    # Generate name
    periods_str = ",".join([f"{p}d" for p in sorted(forward_periods)])
    name = f"{trigger} {condition_desc} → {target} [{periods_str}]"

    experiment = {
        'name': name,
        'trigger_symbol': trigger.upper(),
        'target_symbol': target.upper(),
        'trigger_condition': condition,
        'forward_periods': sorted(forward_periods),
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'enable_ai': enable_ai
    }

    st.session_state.experiments.append(experiment)
    st.success(f"✅ Added: {name}")

def run_all_experiments():
    """Execute all experiments"""
    if not st.session_state.experiments:
        st.error("No experiments to run!")
        return

    st.header("🔬 Results")

    for experiment in st.session_state.experiments:
        with st.container():
            result = st.session_state.scout.run_experiment(experiment)
            display_experiment_result(result, experiment)
            st.divider()

def run_reality_check(experiment, best_period_key, scout):
    """
    Automatically split the full date range into rolling 1-year windows,
    run the signal on each, and surface best/worst/recent results.
    No hardcoded years — adapts to any asset or date range.
    """
    today = datetime.now()

    # Go back up to 6 years from today (or the user's configured start, whichever is further)
    history_start = min(
        pd.to_datetime(experiment['start_date']),
        today - timedelta(days=6 * 365)
    )
    history_end = today

    # Build non-overlapping 1-year windows
    windows = []
    window_start = history_start
    while window_start + timedelta(days=365) <= history_end:
        window_end = window_start + timedelta(days=365)
        windows.append((window_start, window_end))
        window_start += timedelta(days=365)

    if not windows:
        return [], 'default'

    # Run the experiment silently on each window
    window_results = []
    for ws, we in windows:
        test_exp = {**experiment,
                    'start_date': ws.strftime('%Y-%m-%d'),
                    'end_date':   we.strftime('%Y-%m-%d')}
        r = scout.run_experiment(test_exp, silent=True)

        fa = (r or {}).get('forward_analysis', {})
        if not r or r.get('error') or r.get('signals', 0) == 0 \
                or best_period_key not in fa \
                or fa[best_period_key]['valid_signals'] == 0:
            window_results.append({
                'window_start': ws, 'window_end': we,
                'wr': None, 'avg': None, 'signals': 0
            })
            continue

        s = fa[best_period_key]
        window_results.append({
            'window_start': ws, 'window_end': we,
            'wr': s['win_rate'], 'avg': s['avg_return'],
            'signals': s['valid_signals']
        })

    # Pick the most informative windows to show:
    # best year, worst year, most recent year, + any with data
    valid = [w for w in window_results if w['wr'] is not None]
    if not valid:
        return [], 'default'

    best_w   = max(valid, key=lambda w: w['wr'])
    worst_w  = min(valid, key=lambda w: w['wr'])
    recent_w = max(valid, key=lambda w: w['window_start'])

    # Build final display set (deduplicated by window_start)
    seen = set()
    display_windows = []
    for w in [recent_w, best_w, worst_w] + sorted(valid, key=lambda x: x['window_start']):
        key = w['window_start'].year
        if key not in seen:
            seen.add(key)
            display_windows.append(w)

    # Sort chronologically
    display_windows.sort(key=lambda w: w['window_start'])

    rows = []
    for w in display_windows:
        label_parts = [f"{w['window_start'].strftime('%b %Y')} – {w['window_end'].strftime('%b %Y')}"]
        if w is best_w:   label_parts.append('📈 best year')
        if w is worst_w:  label_parts.append('📉 worst year')
        if w is recent_w: label_parts.append('🕐 most recent')
        label = '  '.join(label_parts)

        if w['wr'] is None:
            rows.append({'Period': label, 'Signals': '—', 'Win rate': '—',
                         'Avg gain/loss': '—', 'Holds up?': '⚪ No signals', '_wr': None})
        else:
            wr, avg = w['wr'], w['avg']
            verdict = '✅ Yes' if (wr >= 60 and avg >= 1.0) else ('⚡ Marginal' if wr >= 50 else '❌ No')
            rows.append({
                'Period': label,
                'Signals': w['signals'],
                'Win rate': f"{wr:.0f}%",
                'Avg gain/loss': f"{avg:+.1f}%",
                'Holds up?': verdict,
                '_wr': wr,
            })

    asset_type = classify_asset(format_symbol(experiment['target_symbol']))
    return rows, asset_type


def _signal_verdict(stats, n_signals):
    """Return (label, rec, emoji) based on stats quality"""
    wr  = stats['win_rate']
    avg = stats['avg_return']
    std = stats['std_dev']
    sharpe = (avg / std) if std > 0 else 0

    if wr >= 62 and avg >= 1.5 and n_signals >= 15 and sharpe >= 0.25:
        return 'STRONG SIGNAL', 'consider', '✅'
    elif wr >= 55 and avg >= 0.5 and n_signals >= 10:
        return 'MODERATE SIGNAL', 'caution', '⚡'
    else:
        return 'WEAK SIGNAL', 'skip', '⚠️'


def _quality_score(sharpe):
    if sharpe >= 1.0:  return 'Excellent'
    if sharpe >= 0.5:  return 'Good'
    if sharpe >= 0.3:  return 'Decent'
    if sharpe >= 0.1:  return 'Poor'
    return 'Very Poor'


def _confidence_note(n):
    if n >= 25: return f"{n} signals — solid sample size"
    if n >= 15: return f"{n} signals — reasonable, but more data would help"
    if n >= 8:  return f"{n} signals — limited, treat results with caution"
    return f"{n} signals — too few to trust, results likely random"


def display_experiment_result(result, experiment):
    """Display experiment results in plain English with actionable layout"""
    trigger = experiment['trigger_symbol']
    target  = experiment['target_symbol']

    st.subheader(f"📊 {result['experiment']}")

    if result.get('error'):
        st.error(f"Something went wrong: {result['error']}")
        return

    if result['signals'] == 0:
        st.warning(f"No signals found — {trigger} never met the trigger condition in this date range. Try a lower threshold or a longer lookback period.")
        return

    forward_analysis = result.get('forward_analysis', {})
    if not forward_analysis:
        st.warning("No return data available.")
        return

    # ── Best period ──────────────────────────────────────────────────────────
    best_period = max(
        (p for p, s in forward_analysis.items() if s['valid_signals'] > 0),
        key=lambda p: forward_analysis[p]['avg_return'],
        default=None
    )
    if not best_period:
        st.warning("No valid forward return data.")
        return

    best = forward_analysis[best_period]
    sharpe = (best['avg_return'] / best['std_dev']) if best['std_dev'] > 0 else 0
    verdict_label, rec, verdict_emoji = _signal_verdict(best, best['valid_signals'])

    # ── Verdict banner ───────────────────────────────────────────────────────
    if rec == 'consider':
        st.success(f"{verdict_emoji} **{verdict_label}** — This pattern has a real edge worth acting on.")
    elif rec == 'caution':
        st.warning(f"{verdict_emoji} **{verdict_label}** — There's a slight edge here, but it's thin. Proceed with small size.")
    else:
        st.error(f"{verdict_emoji} **{verdict_label}** — The numbers don't support trading this. Not worth the risk.")

    # ── Plain-English summary ────────────────────────────────────────────────
    wins = round(best['win_rate'] / 100 * best['valid_signals'])
    losses = best['valid_signals'] - wins
    direction_note = "went up" if best['avg_return'] >= 0 else "went down"
    st.markdown(
        f"> Out of **{best['valid_signals']} times** {trigger} triggered, "
        f"{target} {direction_note} over the next {best_period} **{wins} times** and fell **{losses} times** "
        f"({best['win_rate']:.0f}% win rate). "
        f"The average move was **{best['avg_return']:+.2f}%** — "
        f"best trade was **+{best['best_signal']:.1f}%**, worst was **{best['worst_signal']:.1f}%**."
    )
    st.caption(_confidence_note(best['valid_signals']))

    st.divider()

    # ── What happened after the trigger? ────────────────────────────────────
    st.subheader("📋 What happened after the trigger fired?")
    summary_data = []
    for period, stats in forward_analysis.items():
        if stats['valid_signals'] == 0:
            continue
        w = round(stats['win_rate'] / 100 * stats['valid_signals'])
        l = stats['valid_signals'] - w
        summary_data.append({
            'Hold for': period,
            'Times triggered': stats['valid_signals'],
            'Trades that made money': f"{w} of {stats['valid_signals']} ({stats['win_rate']:.0f}%)",
            'Avg profit / loss': f"{stats['avg_return']:+.2f}%",
            'Best trade': f"+{stats['best_signal']:.1f}%",
            'Worst trade': f"{stats['worst_signal']:.1f}%",
            'Total if you traded all': f"{stats['total_return']:+.1f}%",
            'Typical swing': f"±{stats['std_dev']:.1f}%",
        })

    if summary_data:
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    # ── Win rate vs avg return chart ─────────────────────────────────────────
    st.subheader("📊 How did the signal perform over different hold times?")
    periods    = [d['Hold for'] for d in summary_data]
    avg_rets   = [float(d['Avg profit / loss'].replace('%','')) for d in summary_data]
    win_rates  = [float(d['Trades that made money'].split('(')[1].replace('%)','')) for d in summary_data]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=periods, y=avg_rets, name='Avg Profit/Loss %',
        marker_color=['#2ecc71' if r >= 0 else '#e74c3c' for r in avg_rets],
        yaxis='y'
    ))
    fig.add_trace(go.Scatter(
        x=periods, y=win_rates, mode='lines+markers',
        name='% of trades that made money',
        line=dict(color='#f39c12', width=3), marker=dict(size=9),
        yaxis='y2'
    ))
    fig.add_hline(y=50, line_dash='dot', line_color='gray',
                  annotation_text='50% (coin flip)', yref='y2')
    fig.update_layout(
        title=f"Average gain and win rate for each holding period",
        xaxis_title="How long you held after the trigger",
        yaxis=dict(title="Avg Profit / Loss %", side="left"),
        yaxis2=dict(title="% of trades that made money", side="right",
                    overlaying="y", range=[0, 100]),
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Signal timeline ──────────────────────────────────────────────────────
    if result.get('signal_details'):
        st.subheader(f"📅 Every trade, plotted over time ({best_period} hold)")
        signals_df = pd.DataFrame(result['signal_details'])
        signals_df['date'] = pd.to_datetime(signals_df['date'])
        primary_period = next(
            (p for p in ['90d','60d','30d','15d','7d']
             if p in signals_df.columns and signals_df[p].notna().any()), None
        )
        if primary_period:
            valid = signals_df[signals_df[primary_period].notna()].copy()
            valid['Running total'] = valid[primary_period].cumsum()
            valid['Result'] = valid[primary_period].apply(lambda x: 'Profit' if x > 0 else 'Loss')

            fig2 = px.scatter(valid, x='date', y='Running total', color='Result',
                              title=f"Running total profit/loss if you traded every signal ({primary_period} hold)",
                              labels={'Running total': 'Running total profit/loss (%)', 'date': 'When the trigger fired'},
                              color_discrete_map={'Profit': '#2ecc71', 'Loss': '#e74c3c'})
            fig2.add_trace(go.Scatter(
                x=valid['date'], y=valid['Running total'],
                mode='lines', name='Cumulative P&L',
                line=dict(color='royalblue', width=2)
            ))
            fig2.add_hline(y=0, line_dash='dash', line_color='gray',
                           annotation_text='Break even')
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Should I trade this? ─────────────────────────────────────────────────
    st.subheader("🎯 Should I trade this?")

    period_days_val = len(pd.date_range(experiment['start_date'], experiment['end_date']))
    annual_return = (best['avg_return'] * best['valid_signals'] * 365) / period_days_val if period_days_val > 0 else 0
    profit_factor = abs(best['best_signal'] / best['worst_signal']) if best['worst_signal'] != 0 else 0

    if rec == 'consider':
        st.success(f"""
**Verdict: Consider trading this pattern**

When {trigger} triggers, buy {target} and hold for {best_period}.
- You'd have made money on **{best['win_rate']:.0f}%** of trades — roughly **{wins} wins out of {best['valid_signals']}**
- Average gain per trade: **{best['avg_return']:+.2f}%**
- Estimated yearly return if you follow it consistently: **{annual_return:.1f}%**

**How to trade it:** Wait for {trigger} to meet the trigger condition, enter {target} at the next open, exit after {best_period.replace('d',' days')}.
Set a stop-loss around **{abs(best['worst_signal']) / 2:.1f}%** below entry to limit downside.
        """)
    elif rec == 'caution':
        st.warning(f"""
**Verdict: Proceed with caution — thin edge**

The win rate is above a coin flip ({best['win_rate']:.0f}%), but the average gain is small ({best['avg_return']:+.2f}%).
Fees, slippage, and a few bad trades could wipe out the edge entirely.

**If you trade it:** Keep position size small (no more than 2–3% of portfolio per trade).
Only trade it over the {best_period.replace('d',' day')} timeframe — the edge gets weaker beyond that.
        """)
    else:
        best_alt = max(forward_analysis.items(),
                       key=lambda x: x[1]['win_rate'] if x[1]['valid_signals'] > 0 else 0)
        st.error(f"""
**Verdict: Skip this trade**

A **{best['win_rate']:.0f}% win rate** with only **{best['avg_return']:+.2f}% average gain** isn't enough to trade profitably once you account for costs.
The worst single loss was **{best['worst_signal']:.1f}%** — that alone could wipe out many winning trades.

**Try instead:** Lower the trigger threshold, test a longer lookback, or flip the direction (e.g. test what happens when {trigger} *falls* instead of rises).
        """)

    # ── Auto reality check ───────────────────────────────────────────────────
    period_num   = int(best_period.replace('d', ''))
    is_suspicious = best['win_rate'] > 75 or annual_return > 150

    if is_suspicious:
        st.divider()
        st.subheader("🔍 Automatic Reality Check")
        st.caption(
            f"A win rate of **{best['win_rate']:.0f}%** (or an estimated yearly return of "
            f"**{annual_return:.0f}%**) is high enough that it could be a bull-market artifact. "
            f"Re-running the same signal on specific market periods to see if the edge holds up..."
        )
        with st.spinner("Validating across historically relevant stress periods..."):
            rc_rows, asset_type = run_reality_check(experiment, best_period, st.session_state.scout)

        display_rows = [{k: v for k, v in r.items() if not k.startswith('_')} for r in rc_rows]
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

        # Asset-specific caveats
        asset_notes = {
            'commodity_energy':  "Note: 2022 was a *good* year for energy (oil surged on Ukraine war), so that's not used as a stress test here — 2023's selloff and 2020's COVID crash are more relevant.",
            'commodity_metal':   "Note: Commodity metals had mixed 2022 performance (gold flat, copper volatile). The stress tests here reflect periods of actual demand weakness.",
            'commodity_agri':    "Note: Agricultural commodities spiked in 2022 (Ukraine/Russia supply shock), so 2023's retreat and 2019's bear are used instead.",
            'crypto':            "Note: Crypto has its own distinct cycles — 2018 (-80%) and 2022 (-65%) are the two real stress tests. Bull markets in between don't validate the signal.",
            'forex':             "Note: Forex pairs don't have 'bear markets' the same way stocks do — stress tests here reflect high-volatility regimes instead.",
            'intl_developed':    "Note: International developed markets largely track US cycles but with currency effects. A signal on Japanese or European stocks also has yen/euro exposure baked in.",
            'intl_emerging':     "Note: Emerging markets have their own political/currency risks. A high win rate here may depend heavily on which country and what the dollar was doing.",
        }
        if asset_type in asset_notes:
            st.caption(asset_notes[asset_type])

        valid = [r for r in rc_rows if r['_wr'] is not None]
        if valid:
            holds    = sum(1 for r in valid if r['_wr'] >= 60)
            marginal = sum(1 for r in valid if 50 <= r['_wr'] < 60)
            breaks   = sum(1 for r in valid if r['_wr'] < 50)

            # Check specifically if 2022 (bear) broke down
            bear_row = next((r for r in valid if '2022' in r['Period']), None)
            bear_broke = bear_row and bear_row['_wr'] < 50

            if breaks > holds:
                st.error(
                    f"**The signal broke down in {breaks} of {len(valid)} periods tested.** "
                    f"The strong original result was likely inflated by being tested during a favourable market. "
                    f"Downgrade this to a weak signal and do not trade it with real money until it holds up across bear periods."
                )
            elif bear_broke:
                st.warning(
                    f"**The signal failed in the 2022 bear market** (win rate: {bear_row['Win rate']}). "
                    f"It works when markets are rising, but not when they fall — which is when you need it most. "
                    f"Only trade this when the broader market trend is up."
                )
            elif holds == len(valid):
                st.success(
                    f"**The signal held up across all {len(valid)} periods tested**, including the 2022 bear market. "
                    f"This is a strong sign the edge is real and not just a bull-market fluke. "
                    f"Confidence in the original verdict increases."
                )
            else:
                st.warning(
                    f"**Mixed results** — held up in {holds} period(s), broke down or was marginal in {breaks + marginal}. "
                    f"The signal works in some conditions but not others. "
                    f"Only trade it when market conditions are similar to the periods where it performed."
                )
        else:
            st.info("Not enough historical data across the test periods to draw a conclusion.")

    st.divider()

    # ── Key numbers ──────────────────────────────────────────────────────────
    st.subheader("📌 Key Numbers at a Glance")

    if best['win_rate'] > 95:
        st.warning(
            f"⚠️ **{best['win_rate']:.0f}% win rate is suspiciously high.** "
            "This often happens when testing during a strong bull market — any strategy looks great when everything goes up. "
            "Re-run this on 2022 data (a down year) to see if it holds."
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best holding period", best_period,
                  f"{best['avg_return']:+.2f}% avg per trade",
                  help="The holding period that produced the highest average return. This is when to exit.")
    with col2:
        st.metric("Times it triggered", best['valid_signals'],
                  help="How many times the trigger fired in your date range. Under 15 is too few to be confident.")
    with col3:
        st.metric("Trades that made money", f"{best['win_rate']:.0f}%",
                  help="60–80% is strong. Below 55% and the edge is questionable. Above 95% is likely too good to be true.")
    with col4:
        st.metric("Estimated yearly return", f"{annual_return:.1f}%",
                  help="What you'd earn per year if you followed this signal consistently. S&P 500 averages ~10%/year.")

    # ── Risk ─────────────────────────────────────────────────────────────────
    st.subheader("🛡️ Risk Check")
    col1, col2, col3 = st.columns(3)
    with col1:
        quality = _quality_score(sharpe)
        st.metric("Signal quality score", quality,
                  f"({sharpe:.2f} raw)",
                  help="How much return you get for the risk you take. Excellent = great reward vs risk. Very Poor = barely worth the risk.")
    with col2:
        st.metric("Biggest single loss", f"{abs(best['worst_signal']):.1f}%",
                  help="The worst individual trade in your test period. Your stop-loss should be set before this point.")
    with col3:
        st.metric("Biggest win vs biggest loss", f"{profit_factor:.1f}×",
                  help="How many times larger your best win was vs your worst loss. Above 2× means your upside outweighs your downside.")

    # ── AI Analysis ──────────────────────────────────────────────────────────
    if experiment.get('enable_ai', True):
        with st.expander("🤖 AI Deep-Dive Analysis", expanded=True):
            with st.spinner("Analysing pattern..."):
                analysis = get_ai_analysis(
                    result['experiment'], trigger, target, result
                )
            st.write(analysis)
    else:
        st.info("AI analysis is turned off for this experiment.")

    # ── Benchmark ────────────────────────────────────────────────────────────
    st.subheader("📏 Did the signal beat just buying and holding?")
    try:
        target_data, _ = get_price_data(target, experiment['start_date'], experiment['end_date'])
        if target_data is not None and len(target_data) > 1:
            bh = ((float(target_data.iloc[-1]['Close']) - float(target_data.iloc[0]['Close']))
                  / float(target_data.iloc[0]['Close'])) * 100
            best_strat = max(
                (s['total_return'] for s in forward_analysis.values() if s['valid_signals'] > 0),
                default=0
            )
            best_strat_period = max(
                (p for p, s in forward_analysis.items() if s['valid_signals'] > 0),
                key=lambda p: forward_analysis[p]['total_return'], default='—'
            )
            outperf = best_strat - bh

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Buy & hold {target} return".format(target=target),
                          f"{bh:+.1f}%",
                          help=f"If you had just bought {target} at the start and held, this is what you'd have made.")
            with col2:
                st.metric(f"Signal strategy return ({best_strat_period})",
                          f"{best_strat:+.1f}%",
                          help="Total return if you only bought when the signal fired and held for the best period.")
            with col3:
                delta_color = "normal" if outperf >= 0 else "inverse"
                st.metric("Extra return vs buy & hold",
                          f"{outperf:+.1f}%",
                          delta=f"{outperf:+.1f}%",
                          delta_color=delta_color,
                          help="Positive = the signal beat buy & hold. Negative = you'd have been better off just holding.")

            if outperf < 0:
                st.caption(
                    f"The signal strategy returned {best_strat:+.1f}% vs {bh:+.1f}% for buy & hold. "
                    f"You'd have made {abs(outperf):.1f}% more by simply buying {target} and doing nothing."
                )
            else:
                st.caption(
                    f"The signal strategy beat buy & hold by {outperf:.1f}%. "
                    f"The trigger is adding real value on top of just owning {target}."
                )
    except Exception:
        pass

    st.divider()
    run_monte_carlo_simulation(result, experiment)

def run_monte_carlo_simulation(result, experiment):
    """Simulate future price paths — unconditional vs conditioned on trigger signal"""
    st.subheader("🎲 Where could the price go from here?")

    forward_analysis = result.get('forward_analysis', {})
    if not forward_analysis:
        st.info("No forward analysis data available for Monte Carlo simulation.")
        return

    # Pick the best-performing period to anchor conditional simulation stats
    best_period_key = None
    best_avg = -999
    for period, stats in forward_analysis.items():
        if stats['valid_signals'] > 0 and stats['avg_return'] > best_avg:
            best_avg = stats['avg_return']
            best_period_key = period

    if not best_period_key:
        st.info("No valid signals to base simulation on.")
        return

    st.caption(
        "Runs thousands of possible futures using the asset's historical volatility. "
        "Grey = any random day. Coloured = specifically after your trigger fires."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        n_sims = st.number_input(
            "Number of simulations",
            min_value=500, max_value=20000, value=5000,
            help="More simulations = smoother, more reliable results. 5,000 is a good balance.",
            key=f"mc_sims_{result['experiment']}"
        )
    with col2:
        sim_days = st.number_input(
            "How many days to project forward",
            min_value=7, max_value=365, value=90,
            key=f"mc_days_{result['experiment']}"
        )
    with col3:
        nu = st.number_input(
            "Tail-risk sensitivity",
            min_value=2, max_value=30, value=5,
            help="Lower = models more extreme crashes and spikes (like real markets). Keep between 3–7.",
            key=f"mc_nu_{result['experiment']}"
        )

    with st.spinner("Running simulation..."):
        extended_start = (pd.to_datetime(experiment['start_date']) - timedelta(days=30)).strftime('%Y-%m-%d')
        target_data, error = get_price_data(
            experiment['target_symbol'],
            extended_start,
            experiment['end_date']
        )

    if error or target_data is None or len(target_data) < 20:
        st.warning("Insufficient data for Monte Carlo simulation.")
        return

    close_prices = target_data['Close'].values
    current_price = float(close_prices[-1])
    n_sims_int = int(n_sims)
    sim_days_int = int(sim_days)

    # --- Unconditional simulation: GDX's own daily return history ---
    daily_log_returns = np.diff(np.log(close_prices.astype(float)))
    unc_daily_mean = np.mean(daily_log_returns)
    unc_daily_vol  = np.std(daily_log_returns)

    unc_shocks    = unc_daily_mean + unc_daily_vol * np.random.standard_t(df=nu, size=(n_sims_int, sim_days_int))
    unc_log_paths = np.hstack([np.zeros((n_sims_int, 1)), np.cumsum(unc_shocks, axis=1)])
    unc_paths     = current_price * np.exp(unc_log_paths)

    # --- Conditional simulation: calibrated to signal-triggered returns ---
    # Extract the forward returns that occurred specifically after the trigger fired
    signal_details = result.get('signal_details', [])
    cond_raw = [s[best_period_key] / 100 for s in signal_details
                if best_period_key in s and s[best_period_key] is not None]

    has_conditional = len(cond_raw) >= 5
    cond_paths = None

    if has_conditional:
        period_days = int(best_period_key.replace('d', ''))
        # Convert n-period % returns → log returns → daily equivalents
        log_cond = np.log(1 + np.array(cond_raw))
        cond_daily_mean = np.mean(log_cond) / period_days
        cond_daily_vol  = np.std(log_cond)  / np.sqrt(period_days)

        cond_shocks    = cond_daily_mean + cond_daily_vol * np.random.standard_t(df=nu, size=(n_sims_int, sim_days_int))
        cond_log_paths = np.hstack([np.zeros((n_sims_int, 1)), np.cumsum(cond_shocks, axis=1)])
        cond_paths     = current_price * np.exp(cond_log_paths)

    days = np.arange(sim_days_int + 1)

    def percentile_bands(paths):
        return {p: np.percentile(paths, p, axis=0) for p in [5, 25, 50, 75, 95]}

    unc_bands  = percentile_bands(unc_paths)
    cond_bands = percentile_bands(cond_paths) if has_conditional else None

    # --- Fan chart ---
    fig_fan = go.Figure()

    # Unconditional fan (grey)
    fig_fan.add_trace(go.Scatter(x=days, y=unc_bands[95], name='Unconditional 95th',
        line=dict(color='rgba(150,150,150,0.6)', dash='dash'), legendgroup='unc'))
    fig_fan.add_trace(go.Scatter(x=days, y=unc_bands[75], name='Unconditional 75th',
        fill='tonexty', fillcolor='rgba(150,150,150,0.08)',
        line=dict(color='rgba(150,150,150,0.4)'), legendgroup='unc'))
    fig_fan.add_trace(go.Scatter(x=days, y=unc_bands[50], name='Unconditional Median',
        fill='tonexty', fillcolor='rgba(150,150,150,0.08)',
        line=dict(color='rgba(120,120,120,0.9)', width=2, dash='dot'), legendgroup='unc'))
    fig_fan.add_trace(go.Scatter(x=days, y=unc_bands[25], name='Unconditional 25th',
        fill='tonexty', fillcolor='rgba(150,150,150,0.08)',
        line=dict(color='rgba(150,150,150,0.4)'), legendgroup='unc'))
    fig_fan.add_trace(go.Scatter(x=days, y=unc_bands[5], name='Unconditional 5th',
        fill='tonexty', fillcolor='rgba(150,150,150,0.08)',
        line=dict(color='rgba(150,150,150,0.6)', dash='dash'), legendgroup='unc'))

    # Conditional fan (coloured) — only if enough signals
    if has_conditional:
        fig_fan.add_trace(go.Scatter(x=days, y=cond_bands[95], name='After Trigger 95th',
            line=dict(color='rgba(0,180,0,0.8)', dash='dash'), legendgroup='cond'))
        fig_fan.add_trace(go.Scatter(x=days, y=cond_bands[75], name='After Trigger 75th',
            fill='tonexty', fillcolor='rgba(0,200,0,0.12)',
            line=dict(color='rgba(0,160,0,0.6)'), legendgroup='cond'))
        fig_fan.add_trace(go.Scatter(x=days, y=cond_bands[50], name='After Trigger Median',
            fill='tonexty', fillcolor='rgba(30,100,255,0.12)',
            line=dict(color='royalblue', width=2), legendgroup='cond'))
        fig_fan.add_trace(go.Scatter(x=days, y=cond_bands[25], name='After Trigger 25th',
            fill='tonexty', fillcolor='rgba(255,140,0,0.12)',
            line=dict(color='rgba(200,100,0,0.6)'), legendgroup='cond'))
        fig_fan.add_trace(go.Scatter(x=days, y=cond_bands[5], name='After Trigger 5th',
            fill='tonexty', fillcolor='rgba(200,0,0,0.12)',
            line=dict(color='rgba(200,0,0,0.8)', dash='dash'), legendgroup='cond'))

    fig_fan.add_hline(y=current_price, line_dash='dot', line_color='white',
                      annotation_text='Current Price')
    trigger = experiment['trigger_symbol']
    target  = experiment['target_symbol']
    cond_label = f" | Coloured = after {trigger} triggers ({len(cond_raw)} signals, {best_period_key} basis)" if has_conditional else ""
    fig_fan.update_layout(
        title=f"{target} — {sim_days_int}-day price projection  |  Grey = any day{cond_label}",
        xaxis_title="Days from today",
        yaxis_title="Price",
        legend=dict(x=0.01, y=0.99, font=dict(size=11)),
        hovermode='x unified'
    )
    st.plotly_chart(fig_fan, use_container_width=True)

    if has_conditional:
        st.caption(
            f"**Grey band** = what {target} normally does on any random day. "
            f"**Coloured band** = what {target} does specifically after {trigger} triggers "
            f"(based on {len(cond_raw)} historical instances). "
            f"If the coloured band sits higher, the trigger is genuinely predicting something useful."
        )

    # --- Return distribution ---
    unc_final_ret  = ((unc_paths[:, -1]  - current_price) / current_price) * 100
    cond_final_ret = ((cond_paths[:, -1] - current_price) / current_price) * 100 if has_conditional else None

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=unc_final_ret, nbinsx=100, name='Any random day',
        marker_color='rgba(150,150,150,0.5)', opacity=0.7
    ))
    if has_conditional:
        fig_hist.add_trace(go.Histogram(
            x=cond_final_ret, nbinsx=100, name=f'After {trigger} triggers',
            marker_color='rgba(30,100,255,0.5)', opacity=0.7
        ))
    fig_hist.update_layout(
        barmode='overlay',
        title=f"Spread of possible {sim_days_int}-day outcomes — {target}",
        xaxis_title="Return % at end of period",
        yaxis_title="Number of simulations",
        legend=dict(x=0.01, y=0.99)
    )

    ref = cond_final_ret if has_conditional else unc_final_ret
    r5, r50, r95 = np.percentile(ref, [5, 50, 95])
    fig_hist.add_vline(x=0,   line_color='red',       line_dash='solid', annotation_text='Break even')
    fig_hist.add_vline(x=r5,  line_color='orange',    line_dash='dash',  annotation_text=f'Worst 5%: {r5:.1f}%')
    fig_hist.add_vline(x=r50, line_color='royalblue', line_dash='dash',  annotation_text=f'Most likely: {r50:.1f}%')
    fig_hist.add_vline(x=r95, line_color='green',     line_dash='dash',  annotation_text=f'Best 5%: {r95:.1f}%')
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- Summary metrics ---
    active = cond_final_ret if has_conditional else unc_final_ret
    fp5, fp50, fp95 = np.percentile(active, [5, 50, 95])
    prob_positive   = float((active > 0).mean() * 100)
    expected_return = float(np.mean(active))
    ctx = f"after {trigger} triggers" if has_conditional else "on any day"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current price", f"${current_price:.2f}",
            help="Starting point for all simulations.")
    with col2:
        st.metric("Most likely price in {d} days".format(d=sim_days_int),
            f"${current_price * (1 + r50 / 100):.2f}",
            f"{r50:+.1f}%",
            help=f"The median outcome across {n_sims_int:,} simulations ({ctx}). Half of paths ended above this, half below.")
    with col3:
        st.metric("Chance it goes up", f"{prob_positive:.0f}%",
            help=f"Share of {n_sims_int:,} simulations ({ctx}) that ended above today's price.")
    with col4:
        downside = abs(fp5)
        st.metric("Worst realistic loss (1-in-20)",
            f"{downside:.0f}%",
            f"floor ~${current_price * (1 - downside / 100):.2f}",
            delta_color="inverse",
            help=f"Only 1 in 20 simulations ({ctx}) fell further than this. Think of it as a stress-test worst case.")

    # --- Side-by-side comparison ---
    if has_conditional:
        unc_r5, unc_r50, unc_r95 = np.percentile(unc_final_ret, [5, 50, 95])
        unc_prob = float((unc_final_ret > 0).mean() * 100)
        unc_exp  = float(np.mean(unc_final_ret))
        with st.expander(f"📊 Does the {trigger} trigger actually change the outlook?"):
            st.dataframe(pd.DataFrame({
                'Outcome': ['Chance of gain', 'Most likely return (median)',
                            'Average return', 'Worst 5% scenario', 'Best 5% scenario'],
                f'Any random day': [
                    f"{unc_prob:.0f}%", f"{unc_r50:+.1f}%",
                    f"{unc_exp:+.1f}%", f"{unc_r5:+.1f}%", f"{unc_r95:+.1f}%"
                ],
                f'After {trigger} triggers': [
                    f"{prob_positive:.0f}%", f"{r50:+.1f}%",
                    f"{expected_return:+.1f}%", f"{fp5:+.1f}%", f"{fp95:+.1f}%"
                ],
            }), use_container_width=True, hide_index=True)
            diff_prob = prob_positive - unc_prob
            diff_med  = r50 - unc_r50
            if abs(diff_prob) < 3 and abs(diff_med) < 5:
                st.caption("The two columns look very similar — the trigger isn't meaningfully changing the odds.")
            elif diff_prob > 0 and diff_med > 0:
                st.caption(f"After the trigger fires, the chance of gain improves by {diff_prob:+.0f}% and the median outcome shifts by {diff_med:+.1f}%. The signal is adding edge.")
            else:
                st.caption("Mixed results — the trigger helps on some metrics but not others.")

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    pct_returns = np.percentile(active, percentiles)
    pct_prices  = current_price * (1 + pct_returns / 100)
    with st.expander("📋 Full range of outcomes"):
        st.dataframe(pd.DataFrame({
            'Scenario': ['Worst 5%', 'Worst 10%', 'Below average', 'Middle (most likely)',
                         'Above average', 'Best 10%', 'Best 5%'],
            'Return': [f"{r:+.1f}%" for r in pct_returns],
            'Price at end': [f"${p:.2f}" for p in pct_prices],
        }), use_container_width=True, hide_index=True)



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("Please check your environment setup and try again.")
