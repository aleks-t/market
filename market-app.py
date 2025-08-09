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

# Page config with improved styling
st.set_page_config(
    page_title="Market Scout",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/market-scout',
        'Report a bug': "https://github.com/yourusername/market-scout/issues",
        'About': "# Market Scout 📈\nA powerful tool for discovering market patterns and relationships."
    }
)

# Add custom CSS
st.markdown("""
    <style>
    .stApp {
        # max-width: 1200px;
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

                # Get price data for the lookback period
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
                avg_volume = float(
                    volume_data.iloc[date_idx-lookback:date_idx].mean())
                current_volume = float(volume_data.iloc[date_idx])

                return current_volume > (avg_volume * multiplier)

            return False

        except Exception as e:
            st.write(f"Error checking condition: {str(e)}")
            return False

    def calculate_forward_returns(self, target_data, signal_date, forward_periods):
        """Calculate returns at multiple forward periods from signal date - FIXED VERSION"""
        try:
            signal_idx = target_data.index.get_loc(signal_date)
            signal_price = float(target_data.iloc[signal_idx]['Close'])

            returns = {}

            for period in forward_periods:
                # FIXED: Calculate actual trading days forward, not just adding days
                target_idx = signal_idx + period

                # Ensure we don't go beyond available data
                if target_idx < len(target_data):
                    future_price = float(target_data.iloc[target_idx]['Close'])
                    forward_return = (
                        (future_price - signal_price) / signal_price) * 100
                    returns[f'{period}d'] = forward_return
                else:
                    returns[f'{period}d'] = None

            return returns

        except Exception as e:
            st.write(f"Error calculating forward returns: {str(e)}")
            return {}

    def run_experiment(self, experiment):
        """Run experiment with proper forward-looking analysis - FIXED VERSION"""
        try:
            # Create clean progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Fetch trigger data
            status_text.text("📈 Fetching trigger symbol data...")
            progress_bar.progress(25)

            trigger_data, trigger_error = get_price_data(
                experiment['trigger_symbol'],
                experiment['start_date'],
                experiment['end_date']
            )

            if trigger_error:
                st.error(f"❌ Trigger data error: {trigger_error}")
                return self._empty_result(experiment['name'], trigger_error)

            # Fetch target data (extend end date to get forward returns)
            status_text.text("🎯 Fetching target symbol data...")
            progress_bar.progress(50)

            # Extend end date by max forward period + buffer
            max_forward_days = max(experiment['forward_periods'])
            extended_end = pd.to_datetime(
                experiment['end_date']) + timedelta(days=max_forward_days + 50)

            target_data, target_error = get_price_data(
                experiment['target_symbol'],
                experiment['start_date'],
                extended_end.strftime('%Y-%m-%d')
            )

            if target_error:
                st.error(f"❌ Target data error: {target_error}")
                return self._empty_result(experiment['name'], target_error)

            status_text.text("🔍 Scanning for signal patterns...")
            progress_bar.progress(75)

            signals = []

            # Scan through trigger data for signals
            for i in range(len(trigger_data)):
                try:
                    condition_met = self.check_condition(
                        trigger_data,
                        experiment['trigger_condition'],
                        i
                    )

                    if condition_met:
                        signal_date = trigger_data.index[i]

                        # Only process if signal_date exists in target_data
                        if signal_date in target_data.index:

                            # FIXED: Calculate forward returns at multiple periods
                            forward_returns = self.calculate_forward_returns(
                                target_data,
                                signal_date,
                                experiment['forward_periods']
                            )

                            # Only add signal if we have at least one valid forward return
                            if any(v is not None for v in forward_returns.values()):
                                trigger_price = float(
                                    trigger_data.iloc[i]['Close'])
                                target_price = float(
                                    target_data.loc[signal_date]['Close'])

                                signal_data = {
                                    'date': signal_date.strftime('%Y-%m-%d'),
                                    'trigger_price': trigger_price,
                                    'target_price': target_price,
                                    **forward_returns  # This should now have different values for each period
                                }

                                signals.append(signal_data)

                except Exception as e:
                    continue

            status_text.text("📊 Calculating performance metrics...")
            progress_bar.progress(100)

            # FIXED: Calculate summary statistics for each forward period separately
            results = {
                'experiment': experiment['name'],
                'trigger_symbol': experiment['trigger_symbol'],
                'target_symbol': experiment['target_symbol'],
                'signals': len(signals),
                'signal_details': signals,
                'forward_analysis': {}
            }

            # FIXED: Analyze each forward period independently
            for period in experiment['forward_periods']:
                period_key = f'{period}d'

                # Get only the returns for THIS specific period
                valid_returns = []
                for signal in signals:
                    if period_key in signal and signal[period_key] is not None:
                        valid_returns.append(signal[period_key])

                if valid_returns:
                    # Calculate metrics for THIS period only
                    valid_returns_array = np.array(valid_returns)

                    results['forward_analysis'][period_key] = {
                        'valid_signals': len(valid_returns),
                        'avg_return': float(np.mean(valid_returns_array)),
                        'win_rate': float((valid_returns_array > 0).mean() * 100),
                        'best_signal': float(np.max(valid_returns_array)),
                        'worst_signal': float(np.min(valid_returns_array)),
                        'std_dev': float(np.std(valid_returns_array)),
                        'total_return': float(np.sum(valid_returns_array))
                    }
                else:
                    results['forward_analysis'][period_key] = {
                        'valid_signals': 0,
                        'avg_return': 0,
                        'win_rate': 0,
                        'best_signal': 0,
                        'worst_signal': 0,
                        'std_dev': 0,
                        'total_return': 0
                    }

            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)

            # Clear status after a moment
            import time
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()

            return results

        except Exception as e:
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


def format_symbol(symbol):
    """Format symbol for Yahoo Finance"""
    symbol = symbol.upper().strip()

    # Handle special cases
    symbol_mapping = {
        'BTC': 'BTC-USD',
        'ETH': 'ETH-USD',
        'DXY': 'DX-Y.NYB',
        'VIX': '^VIX',
        'TNX': '^TNX',
        'GOLD': 'GC=F',
        'OIL': 'CL=F'
    }

    return symbol_mapping.get(symbol, symbol)


def get_price_data(symbol, start_date, end_date):
    """Fetch price data with robust error handling"""
    try:
        formatted_symbol = format_symbol(symbol)

        # Add retry logic for data fetching
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

            except Exception as e:
                if attempt == 2:  # Last attempt
                    return None, f"Failed to fetch data for {symbol} after 3 attempts: {str(e)}"
                continue

        if data.empty:
            return None, f"No data available for {symbol} ({formatted_symbol})"

        # Handle multi-level columns more robustly
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Handle the case where all columns have the same name
        if len(data.columns) >= 4 and len(set(data.columns)) == 1:
            expected_names = ['Open', 'High', 'Low',
                              'Close', 'Adj Close', 'Volume']
            actual_names = expected_names[:len(data.columns)]
            data.columns = actual_names

        # Try different possible column names for Close price
        close_column = None
        for col_name in ['Close', 'Adj Close', 'close', 'adj_close']:
            if col_name in data.columns:
                close_column = col_name
                break

        if close_column is None:
            return None, f"No Close price column found for {symbol}. Available columns: {list(data.columns)}"

        # Rename to standard 'Close' for consistency
        if close_column != 'Close':
            data = data.rename(columns={close_column: 'Close'})

        # Ensure we have Volume column
        if 'Volume' not in data.columns:
            data['Volume'] = 1000000

        # Remove any rows with NaN values in Close column
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
        # Get API key from environment variable
        api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not api_key:
            return "❌ AI analysis disabled: ANTHROPIC_API_KEY environment variable not set."

        client = anthropic.Anthropic(api_key=api_key)

        # Get the primary analysis period (longest one with data)
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
    # Header - Clean and simple
    st.title("🔍 Market Scout")

    # Sidebar
    with st.sidebar:
        st.header("Create Experiment")

        # Quick presets - more compact
        st.write("**📚 Popular Patterns**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("GLD → GDX", use_container_width=True):
                load_preset('gld-gdx')
            if st.button("VIX → SPY", use_container_width=True):
                load_preset('vix-spy')
            if st.button("BTC → NASDAQ", use_container_width=True):
                load_preset('btc-nasdaq')

        with col2:
            if st.button("DXY → Gold", use_container_width=True):
                load_preset('dxy-gold')
            if st.button("IWM → SPY", use_container_width=True):
                load_preset('iwm-spy')
            if st.button("Oil → Energy", use_container_width=True):
                load_preset('oil-energy')

        st.divider()

        # Experiment form - enhanced with tooltips
        trigger_symbol = st.text_input(
            "Trigger Symbol",
            value=st.session_state.get('trigger_symbol', ''),
            help="📡 The asset that creates the signal (e.g., GLD, VIX, BTC). Use standard tickers. Crypto: BTC, ETH. Indices: VIX, DXY, TNX. Commodities: GOLD, OIL"
        )

        target_symbol = st.text_input(
            "Target Symbol",
            value=st.session_state.get('target_symbol', ''),
            help="🎯 The asset whose returns we measure (e.g., GDX, SPY, QQQ). This is what you would actually trade. Should be related to trigger symbol."
        )

        # Trigger condition - enhanced tooltips
        st.write("**🎯 Trigger Condition**")
        condition_type = st.selectbox(
            "Condition Type",
            ["Price Change", "Above Moving Average", "Volume Spike"],
            help="Price Change: When X moves Y%, what happens to Z? (Most popular). Above MA: Trend breakouts. Volume Spike: Unusual activity detection."
        )

        if condition_type == "Price Change":
            col1, col2 = st.columns(2)
            with col1:
                direction = st.selectbox(
                    "Direction", ["Up", "Down"],
                    help="Up: Trigger when price increases. Down: Trigger when price decreases. Test both directions!"
                )
            with col2:
                threshold = st.number_input(
                    "Threshold %", value=2.0, min_value=0.1, step=0.1,
                    help="Minimum price move to trigger signal. 1-3%: More signals. 3-5%: Balanced. 5%+: Fewer but stronger."
                )
            lookback_days = st.number_input(
                "Lookback Days", value=1, min_value=1, max_value=10,
                help="Time period for measuring price change. 1 day: Immediate reactions (most common). 2-3 days: Short momentum."
            )

        elif condition_type == "Above Moving Average":
            ma_period = st.number_input(
                "MA Period", value=50, min_value=5, max_value=200,
                help="Number of days for MA calculation. 20 days: Short-term. 50 days: Medium-term (most popular). 200 days: Long-term."
            )

        elif condition_type == "Volume Spike":
            col1, col2 = st.columns(2)
            with col1:
                vol_lookback = st.number_input(
                    "Lookback Days", value=20, min_value=5,
                    help="Days to calculate average volume. 10 days: Recent. 20 days: Monthly average (standard). 60 days: Long-term."
                )
            with col2:
                vol_multiplier = st.number_input(
                    "Volume Multiplier", value=2.0, min_value=1.1, step=0.1,
                    help="How much above average = spike. 1.5x: Moderate. 2.0x: Significant (recommended). 3.0x: Rare events."
                )

        # Analysis period - enhanced tooltips
        st.write("**📅 Analysis Period**")
        col1, col2 = st.columns(2)
        with col1:
            days_back = st.number_input(
                "Days Back", value=365, min_value=90, max_value=1825,
                help="How far back to look for patterns. 365 days: Recent patterns. 730 days: Balanced. 1095+ days: More signals but may be outdated."
            )
        with col2:
            end_date = st.date_input(
                "End Date", value=datetime.now().date(),
                help="When to stop looking for signals. Use today's date for most recent analysis."
            )

        start_date = end_date - timedelta(days=days_back)

        # Enhanced date range display with signal count estimate
        # Approximate trading days (weekends excluded)
        trading_days = days_back * 0.71
        st.caption(
            f"📊 **Date Range**: {start_date.strftime('%m/%d/%y')} to {end_date.strftime('%m/%d/%y')}\n"
            f"📈 **~{trading_days:.0f} trading days** | Expect **5-50 signals** for most patterns"
        )

        # Forward periods - enhanced tooltips
        st.write("**📈 Forward Analysis Periods**")
        st.caption("⏱️ How long to hold after each signal")

        col1, col2 = st.columns(2)
        with col1:
            analyze_7d = st.checkbox(
                "7 days", value=True,
                help="Quick momentum plays. Best for day/swing trading. Captures immediate reactions."
            )
            analyze_30d = st.checkbox(
                "30 days", value=True,
                help="Monthly momentum. Best for position trading. Captures sustained moves."
            )
            analyze_90d = st.checkbox(
                "90 days", value=False,
                help="Quarterly trends. Best for long-term investing. Higher risk but potentially bigger returns."
            )

        with col2:
            analyze_15d = st.checkbox(
                "15 days", value=False,
                help="Short-term momentum. Good for swing trading. Captures post-earnings moves."
            )
            analyze_60d = st.checkbox(
                "60 days", value=False,
                help="Medium-term positions. Captures sector rotations and policy changes."
            )
            analyze_180d = st.checkbox(
                "180 days", value=False,
                help="Long-term investing. Captures major market cycles. Highest risk/reward."
            )

        # Build forward periods list
        forward_periods = []
        if analyze_7d:
            forward_periods.append(7)
        if analyze_15d:
            forward_periods.append(15)
        if analyze_30d:
            forward_periods.append(30)
        if analyze_60d:
            forward_periods.append(60)
        if analyze_90d:
            forward_periods.append(90)
        if analyze_180d:
            forward_periods.append(180)

        # Analysis options - enhanced
        enable_ai = st.checkbox(
            "🤖 Enable AI Analysis", value=True,
            help="Get Adam Robinson-style trading insights including practical recommendations, risk management advice, and strategy optimization tips."
        )

        # Add experiment
        if st.button("Add Experiment", use_container_width=True, type="primary"):
            add_experiment(trigger_symbol, target_symbol, condition_type, direction if condition_type == "Price Change" else None,
                           threshold if condition_type == "Price Change" else None,
                           lookback_days if condition_type == "Price Change" else None,
                           ma_period if condition_type == "Above Moving Average" else None,
                           vol_lookback if condition_type == "Volume Spike" else None,
                           vol_multiplier if condition_type == "Volume Spike" else None,
                           forward_periods, start_date, end_date, enable_ai)

    # Main content
    if st.session_state.experiments:
        st.header("🧪 Experiments")

        # Show experiments with enhanced info
        for i, exp in enumerate(st.session_state.experiments):
            with st.expander(f"📊 {exp['name']}", expanded=False):
                col1, col2, col3 = st.columns([3, 3, 1])
                with col1:
                    st.write(f"**Trigger:** {exp['trigger_symbol']}")
                    st.write(f"**Target:** {exp['target_symbol']}")
                with col2:
                    st.write(
                        f"**Period:** {exp['start_date']} to {exp['end_date']}")
                    ai_status = "🤖 AI Enabled" if exp.get(
                        'enable_ai', True) else "📊 AI Disabled"
                    st.write(f"**Analysis:** {ai_status}")
                    st.write(
                        f"**Forward Analysis:** {', '.join([f'{p}d' for p in exp['forward_periods']])}")
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
            
            ### **Rule #1: Start with Economic Logic, Not Random Testing**
            **❌ Don't do**: Test AAPL vs random symbols hoping to find patterns  
            **✅ Do**: Ask "What SHOULD be connected?" then test if it's true
            
            ### **Rule #2: Follow the Money Flows**
            **Ask yourself**: "When investors get excited about X, what else do they buy?"
            
            **Examples:**
            - **Clean Energy Hype**: TSLA ↗️ → Solar stocks, uranium, copper
            - **AI Boom**: NVDA ↗️ → Cloud storage, data centers, semiconductors  
            - **Inflation Fear**: Gold ↗️ → TIPS, mining stocks, real estate
            
            ### **Rule #3: Look for Leading Indicators**
            **What happens FIRST, then influences everything else?**
            
            **Market Leaders:**
            - **VIX** (fear) → Everything else
            - **Dollar (DXY)** → International markets
            - **10-Year Treasury (TNX)** → All interest-sensitive sectors
            - **Oil** → Transportation, inflation expectations
            - **Small Caps (IWM)** → Risk appetite
            
            ### **Rule #4: Study Sector Rotations**
            **When growth is hot** → QQQ ↗️, what happens to:
            - Value stocks (IWV)
            - Utilities (XLU)
            - REITs (VNQ)
            - Bonds (TLT)
            
            ### **Rule #5: Geographic Arbitrage**
            **US markets strong** → What about:
            - Emerging Markets (EEM)
            - Europe (EFA)
            - China (ASHR)
            - Currency effects
            
            ### **Rule #6: Defensive vs Risk-On Patterns**
            - **VIX spikes** → Safe havens (TLT, GLD, Utilities)
            - **Credit spreads widen** → High-quality stocks outperform
            - **Dollar strengthens** → US exporters vs importers
            
            ### **Rule #7: Commodity Chains**
            - **Copper rises** → Industrial stocks, infrastructure
            - **Lumber spikes** → Homebuilders vs mortgage REITs
            - **Natural gas** → Utilities vs chemical companies
            
            ### **Rule #8: Sentiment Extremes**
            - **FOMO peaks** (meme stocks rally) → What crashes next?
            - **Maximum pessimism** → What bounces first?
            
            ### **Rule #9: Challenge Conventional Wisdom**
            **Everyone believes**: "Tech leads the market"  
            **Test**: Does QQQ actually predict SPY, or vice versa?
            
            **Everyone believes**: "Gold is an inflation hedge"  
            **Test**: GLD vs actual inflation data vs inflation expectations
            
            ### **Rule #10: Look for Broken Relationships**
            **Used to work**: Small caps led large caps  
            **Test now**: Does IWM → SPY still work post-2020?
            
            ---
            
            ## 🚀 **Your Systematic Testing Phases**
            
            ### **Phase 1: Map the Obvious** *(Start Here)*
            1. **VIX → SPY** (fear → markets)
            2. **DXY → EEM** (dollar → emerging markets)
            3. **TNX → TLT** (rates → bonds)
            4. **USO → XLE** (oil → energy sector)
            
            ### **Phase 2: Second-Order Effects** *(After Phase 1)*
            1. **TSLA → Clean energy complex** (ICLN, URA, FCX)
            2. **NVDA → AI ecosystem** (CLOU, ROBO, SMH)
            3. **Housing data → REIT sectors** (VNQ, REZ, IYR)
            4. **BTC → Risk assets** (QQQ, IWM, EEM)
            
            ### **Phase 3: Contrarian Ideas** *(Advanced)*
            1. **High VIX → Buy small caps** (when fear peaks, risk appetite returns)
            2. **Bond crashes → Buy growth** (falling bonds = rising rates = growth premium)
            3. **Extreme sentiment → Fade the move** (test mean reversion)
            
            ---
            
            ## 📊 **Quick Reference: Top 10 Tests to Start With**
            
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
            
            ### ⚠️ **What Makes a Good Pattern:**
            - ✅ Economic logic explaining WHY it should work
            - ✅ 50+ signals for statistical significance
            - ✅ 60-80% win rate (realistic)
            - ✅ Works across different market conditions
            - ✅ Shorter holding periods (7-30 days)
            
            ### ❌ **Avoid These Beginner Mistakes:**
            - Testing unrelated assets (AAPL → UNG)
            - Looking for 100% win rates (impossible)
            - Testing during bull markets only
            - Using holding periods >90 days (too much noise)
            """)

        # Help section
        with st.expander("💡 How This Works"):
            st.write("""
            **Enhanced Implementation:**
            
            1. **🔍 Finds Trigger Signals**: Scans for when your trigger symbol meets the condition (e.g., GLD moves 2%+)
            
            2. **📈 Measures Forward Returns**: For each signal, calculates what the target symbol returns 7, 30, 60, 90+ days later
            
            3. **📊 Statistical Analysis**: Shows win rates, average returns, and risk metrics for each time period
            
            4. **🤖 AI Insights**: Provides Adam Robinson-style trading strategy recommendations based on the results
            
            5. **✨ Enhanced Features**: Better tooltips, cleaner progress tracking, and improved visual feedback
            
            **Example**: If GLD spikes 2% on Monday, what does GDX return by Friday (7d), next month (30d), etc.
            """)


def load_preset(preset_name):
    """Load preset experiment configurations"""
    presets = {
        'gld-gdx': {
            'trigger': 'GLD',
            'target': 'GDX',
            'threshold': 1.5,
            'lookback': 1,
            'days_back': 730
        },
        'vix-spy': {
            'trigger': '^VIX',
            'target': 'SPY',
            'threshold': 15,
            'lookback': 1,
            'days_back': 1095
        },
        'btc-nasdaq': {
            'trigger': 'BTC-USD',
            'target': 'QQQ',
            'threshold': 5,
            'lookback': 1,
            'days_back': 1095
        },
        'dxy-gold': {
            'trigger': 'DX-Y.NYB',
            'target': 'GLD',
            'threshold': 1,
            'lookback': 1,
            'days_back': 1095
        },
        'iwm-spy': {
            'trigger': 'IWM',
            'target': 'SPY',
            'threshold': 2,
            'lookback': 1,
            'days_back': 1095
        },
        'oil-energy': {
            'trigger': 'USO',
            'target': 'XLE',
            'threshold': 3,
            'lookback': 1,
            'days_back': 730
        }
    }

    if preset_name in presets:
        p = presets[preset_name]
        st.session_state.trigger_symbol = p['trigger']
        st.session_state.target_symbol = p['target']
        st.rerun()


def add_experiment(trigger, target, condition_type, direction, threshold, lookback, ma_period, vol_lookback, vol_multiplier, forward_periods, start_date, end_date, enable_ai=True):
    """Add experiment to session state"""
    if not all([trigger, target]) or not forward_periods:
        st.error(
            "Please fill in trigger symbol, target symbol, and select at least one forward period")
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
    if enable_ai:
        st.info("🤖 AI analysis will be generated when experiment runs")
    else:
        st.info("📊 Experiment added without AI analysis")


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


def display_experiment_result(result, experiment):
    """Display comprehensive experiment results"""
    st.subheader(f"📊 {result['experiment']}")

    if result.get('error'):
        st.error(f"❌ {result['error']}")
        return

    if result['signals'] == 0:
        st.warning("⚠️ No signals found in this period")
        return

    # Multi-timeframe results table
    st.subheader("📈 Forward Return Analysis")

    forward_analysis = result.get('forward_analysis', {})

    if forward_analysis:
        # Create summary table
        summary_data = []
        for period, stats in forward_analysis.items():
            if stats['valid_signals'] > 0:
                summary_data.append({
                    'Period': period,
                    'Signals': stats['valid_signals'],
                    'Avg Return': f"{stats['avg_return']:.2f}%",
                    'Win Rate': f"{stats['win_rate']:.1f}%",
                    'Best': f"{stats['best_signal']:.2f}%",
                    'Worst': f"{stats['worst_signal']:.2f}%",
                    'Total Return': f"{stats['total_return']:.2f}%",
                    'Risk (Std)': f"{stats['std_dev']:.2f}%"
                })

        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)

            # Visualization of forward returns
            st.subheader("📊 Performance by Time Period")

            periods = [data['Period'] for data in summary_data]
            avg_returns = [float(data['Avg Return'].strip('%'))
                           for data in summary_data]
            win_rates = [float(data['Win Rate'].strip('%'))
                         for data in summary_data]

            fig = go.Figure()

            # Add average returns
            fig.add_trace(go.Bar(
                x=periods,
                y=avg_returns,
                name='Avg Return %',
                marker_color='lightblue',
                yaxis='y'
            ))

            # Add win rate line
            fig.add_trace(go.Scatter(
                x=periods,
                y=win_rates,
                mode='lines+markers',
                name='Win Rate %',
                line=dict(color='red', width=3),
                marker=dict(size=8),
                yaxis='y2'
            ))

            fig.update_layout(
                title="Average Returns vs Win Rates by Time Period",
                xaxis_title="Time Period",
                yaxis=dict(title="Average Return %", side="left"),
                yaxis2=dict(title="Win Rate %", side="right", overlaying="y"),
                legend=dict(x=0.01, y=0.99),
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Signal timeline
            if result.get('signal_details'):
                st.subheader("📅 Signal Timeline")

                signals_df = pd.DataFrame(result['signal_details'])
                signals_df['date'] = pd.to_datetime(signals_df['date'])

                # Choose primary period for timeline (longest available)
                primary_period = None
                for period in ['90d', '60d', '30d', '15d', '7d']:
                    if period in signals_df.columns and signals_df[period].notna().any():
                        primary_period = period
                        break

                if primary_period:
                    # Create cumulative return chart
                    valid_signals = signals_df[signals_df[primary_period].notna()].copy(
                    )
                    valid_signals['cumulative_return'] = valid_signals[primary_period].cumsum(
                    )
                    valid_signals['trade_result'] = valid_signals[primary_period].apply(
                        lambda x: 'Win' if x > 0 else 'Loss')

                    fig = px.scatter(valid_signals,
                                     x='date',
                                     y='cumulative_return',
                                     color='trade_result',
                                     title=f"Cumulative Returns Over Time ({primary_period} holding period)",
                                     labels={'cumulative_return': f'Cumulative Return % ({primary_period})',
                                             'date': 'Signal Date'},
                                     color_discrete_map={'Win': 'green', 'Loss': 'red'})

                    # Add cumulative line
                    fig.add_trace(go.Scatter(
                        x=valid_signals['date'],
                        y=valid_signals['cumulative_return'],
                        mode='lines',
                        name='Cumulative Return',
                        line=dict(color='blue', width=2)
                    ))

                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig, use_container_width=True)

            # Get best performing period
            best_period = None
            best_return = -999
            for period, stats in forward_analysis.items():
                if stats['valid_signals'] > 0 and stats['avg_return'] > best_return:
                    best_return = stats['avg_return']
                    best_period = period

            # Trading Stats Summary
            st.subheader("💼 Trading Summary")

            # Reality check warning for 100% win rates
            if best_period:
                best_stats = forward_analysis[best_period]

                if best_stats['win_rate'] > 95:
                    st.warning(f"""
                    ⚠️ **Reality Check**: {best_stats['win_rate']:.0f}% win rate is extremely suspicious!
                    
                    **Why this is likely misleading:**
                    • Testing during 2024-2025 bull market (+25% both years)
                    • Long holding periods mask true signal strength
                    • Any random strategy would look good in this period
                    
                    **To verify this pattern:**
                    • Test during 2022 bear market (-18% SPY return)
                    • Use shorter periods (7-30 days, not {best_period})
                    • Compare to random entry dates
                    """)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Best Time Period",
                        best_period,
                        f"{best_stats['avg_return']:.2f}% avg",
                        help="🏆 The holding period (7d, 30d, etc.) that produced the highest average returns. This is your optimal exit timing for this pattern."
                    )
                with col2:
                    st.metric(
                        "Total Signals",
                        best_stats['valid_signals'],
                        help="📊 Number of times this pattern triggered during analysis. More signals = more statistical confidence, but need 20+ for reliability."
                    )
                with col3:
                    st.metric(
                        "Win Rate",
                        f"{best_stats['win_rate']:.1f}%",
                        help="🎯 Percentage of trades that were profitable. 60-80% is excellent. Above 95% is usually too good to be true."
                    )
                with col4:
                    annual_return = (best_stats['avg_return'] * best_stats['valid_signals'] * 365) / (
                        len(pd.date_range(experiment['start_date'], experiment['end_date'])))
                    st.metric(
                        "Annualized Return",
                        f"{annual_return:.1f}%",
                        help="📈 Expected yearly return if you followed this strategy consistently. Compare to S&P 500 (~10% annually). Above 20% is very good."
                    )

                # Risk-adjusted metrics
                st.subheader("📊 Risk Metrics")
                col1, col2, col3 = st.columns(3)

                with col1:
                    sharpe_approx = best_stats['avg_return'] / \
                        best_stats['std_dev'] if best_stats['std_dev'] > 0 else 0
                    st.metric("Sharpe Ratio (approx)", f"{sharpe_approx:.2f}",
                              help="Risk-adjusted return measure. >0.5 is decent, >1.0 is great, >2.0 is excellent. Higher = better returns for the risk taken.")

                with col2:
                    max_loss = abs(best_stats['worst_signal'])
                    st.metric("Max Single Loss", f"{max_loss:.2f}%",
                              help="The worst single trade result - your maximum possible loss on any one trade. This is your worst-case scenario.")

                with col3:
                    profit_factor = abs(best_stats['best_signal'] / best_stats['worst_signal']
                                        ) if best_stats['worst_signal'] != 0 else float('inf')
                    st.metric("Best/Worst Ratio", f"{profit_factor:.2f}",
                              help="Best trade ÷ Worst trade. Shows if your wins are bigger than your losses. >2.0 means wins are twice as big as losses.")

                # Show details about worst loss timing
                if result.get('signal_details'):
                    signals_df = pd.DataFrame(result['signal_details'])
                    if best_period in signals_df.columns:
                        worst_trade_idx = signals_df[best_period].idxmin()
                        worst_trade = signals_df.iloc[worst_trade_idx]
                        st.caption(
                            f"📅 Worst loss ({max_loss:.2f}%) occurred on signal date: {worst_trade['date']}")

                # FIXED: Debug info to verify calculations are different
                with st.expander("🔍 Debug: Verify Different Calculations", expanded=False):
                    st.write("**Forward Analysis Results by Period:**")
                    for period, stats in forward_analysis.items():
                        if stats['valid_signals'] > 0:
                            st.write(
                                f"**{period}**: {stats['valid_signals']} signals, {stats['avg_return']:.3f}% avg return, {stats['win_rate']:.2f}% win rate")

                    if result.get('signal_details'):
                        st.write("**Sample Signal Details (first 3):**")
                        sample_signals = result['signal_details'][:3]
                        for i, signal in enumerate(sample_signals):
                            st.write(f"Signal {i+1} ({signal['date']}):")
                            for period in experiment['forward_periods']:
                                period_key = f'{period}d'
                                if period_key in signal:
                                    st.write(
                                        f"  - {period_key}: {signal[period_key]:.3f}% return")

            # AI Analysis - enhanced with better loading indicator
            if experiment.get('enable_ai', True):
                with st.expander("🎓 AI Analysis & Trading Strategy", expanded=True):
                    # Generate analysis with simple loading
                    with st.spinner("Generating AI analysis..."):
                        analysis = get_ai_analysis(
                            result['experiment'],
                            experiment['trigger_symbol'],
                            experiment['target_symbol'],
                            result
                        )

                    # Display results
                    st.write(analysis)
            else:
                st.info("🤖 AI Analysis disabled for this experiment")

            # Benchmark comparison
            create_benchmark_comparison(result, experiment)

        else:
            st.warning("No forward return data available for visualization")


def create_benchmark_comparison(result, experiment):
    """Create benchmark comparison for the pattern"""
    st.subheader("🎯 Benchmark Comparison")

    # Simple benchmark: buy and hold target symbol
    try:
        target_data, _ = get_price_data(
            experiment['target_symbol'],
            experiment['start_date'],
            experiment['end_date']
        )

        if target_data is not None and len(target_data) > 1:
            # Calculate buy-and-hold return
            start_price = target_data.iloc[0]['Close']
            end_price = target_data.iloc[-1]['Close']
            buy_hold_return = ((end_price - start_price) / start_price) * 100

            # Get best strategy return
            forward_analysis = result.get('forward_analysis', {})
            best_strategy_return = 0
            best_period = None

            for period, stats in forward_analysis.items():
                if stats['valid_signals'] > 0:
                    strategy_return = stats['total_return']
                    if strategy_return > best_strategy_return:
                        best_strategy_return = strategy_return
                        best_period = period

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Buy & Hold Return", f"{buy_hold_return:.2f}%")
            with col2:
                st.metric(
                    f"Strategy Return ({best_period})", f"{best_strategy_return:.2f}%")
            with col3:
                outperformance = best_strategy_return - buy_hold_return
                st.metric("Outperformance", f"{outperformance:.2f}%",
                          delta=f"{outperformance:.2f}%")

    except Exception as e:
        st.caption(f"Benchmark comparison unavailable: {str(e)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("Please check your environment setup and try again.")
