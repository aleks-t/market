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

    def run_experiment(self, experiment):
        """Run experiment with proper forward-looking analysis - FIXED VERSION"""
        try:
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

            max_forward_days = max(experiment['forward_periods'])
            extended_end = pd.to_datetime(experiment['end_date']) + timedelta(days=max_forward_days + 50)

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

                        if signal_date in target_data.index:
                            forward_returns = self.calculate_forward_returns(
                                target_data,
                                signal_date,
                                experiment['forward_periods']
                            )

                            # Only add signal if we have at least one valid forward return
                            if any(v is not None for v in forward_returns.values()):
                                trigger_price = float(trigger_data.iloc[i]['Close'])
                                target_price = float(target_data.loc[signal_date]['Close'])

                                signal_data = {
                                    'date': signal_date.strftime('%Y-%m-%d'),
                                    'trigger_price': trigger_price,
                                    'target_price': target_price,
                                    **forward_returns
                                }

                                signals.append(signal_data)

                except Exception:
                    continue

            status_text.text("📊 Calculating performance metrics...")
            progress_bar.progress(100)

            # Calculate summary statistics for each forward period separately
            results = {
                'experiment': experiment['name'],
                'trigger_symbol': experiment['trigger_symbol'],
                'target_symbol': experiment['target_symbol'],
                'signals': len(signals),
                'signal_details': signals,
                'forward_analysis': {}
            }

            # Analyze each forward period independently
            for period in experiment['forward_periods']:
                period_key = f'{period}d'

                # Get only the returns for THIS specific period
                valid_returns = []
                for signal in signals:
                    if period_key in signal and signal[period_key] is not None:
                        valid_returns.append(signal[period_key])

                if valid_returns:
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

        # Experiment form
        trigger_symbol = st.text_input(
            "Trigger Symbol",
            value=st.session_state.get('trigger_symbol', ''),
            help="📡 The asset that creates the signal (e.g., GLD, VIX, BTC)"
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

        # Help section
        with st.expander("💡 How This Works"):
            st.write("""
            1. **🔍 Finds Trigger Signals**: Scans for when your trigger symbol meets the condition
            2. **📈 Measures Forward Returns**: For each signal, calculates target symbol returns
            3. **📊 Statistical Analysis**: Shows win rates, average returns, and risk metrics
            4. **🤖 AI Insights**: Provides trading strategy recommendations
            
            **Example**: If GLD spikes 2% on Monday, what does GDX return by Friday (7d), next month (30d), etc.
            """)

def load_preset(preset_name):
    """Load preset experiment configurations"""
    presets = {
        'gld-gdx': {'trigger': 'GLD', 'target': 'GDX'},
        'vix-spy': {'trigger': '^VIX', 'target': 'SPY'},
        'btc-nasdaq': {'trigger': 'BTC-USD', 'target': 'QQQ'},
        'dxy-gold': {'trigger': 'DX-Y.NYB', 'target': 'GLD'},
        'iwm-spy': {'trigger': 'IWM', 'target': 'SPY'},
        'oil-energy': {'trigger': 'USO', 'target': 'XLE'}
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
        title=f"{target} — Next {sim_days_int} Days  |  Grey = any day{cond_label}",
        xaxis_title="Days Ahead",
        yaxis_title="Price",
        legend=dict(x=0.01, y=0.99, font=dict(size=11)),
        hovermode='x unified'
    )
    st.plotly_chart(fig_fan, use_container_width=True)

    if has_conditional:
        st.caption(
            f"**Grey** = unconditional (GDX on any random day).  "
            f"**Coloured** = calibrated to the {len(cond_raw)} historical instances where "
            f"{trigger} triggered this signal — these are the paths *most relevant to your trade*. "
            f"If the coloured fan sits higher than the grey, the trigger adds real edge."
        )

    # --- Return distribution histogram ---
    unc_final_ret  = ((unc_paths[:, -1]  - current_price) / current_price) * 100
    cond_final_ret = ((cond_paths[:, -1] - current_price) / current_price) * 100 if has_conditional else None

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=unc_final_ret, nbinsx=100, name='Unconditional',
        marker_color='rgba(150,150,150,0.5)', opacity=0.7
    ))
    if has_conditional:
        fig_hist.add_trace(go.Histogram(
            x=cond_final_ret, nbinsx=100, name=f'After {trigger} Trigger',
            marker_color='rgba(30,100,255,0.5)', opacity=0.7
        ))
    fig_hist.update_layout(
        barmode='overlay',
        title=f"Distribution of {sim_days_int}-Day Returns — {target}",
        xaxis_title="Return %", yaxis_title="Scenarios",
        legend=dict(x=0.01, y=0.99)
    )

    # Reference lines on the conditional if available, else unconditional
    ref = cond_final_ret if has_conditional else unc_final_ret
    r5, r50, r95 = np.percentile(ref, [5, 50, 95])
    fig_hist.add_vline(x=0,   line_color='red',       line_dash='solid', annotation_text='Breakeven')
    fig_hist.add_vline(x=r5,  line_color='orange',    line_dash='dash',  annotation_text=f'5th: {r5:.1f}%')
    fig_hist.add_vline(x=r50, line_color='royalblue', line_dash='dash',  annotation_text=f'Median: {r50:.1f}%')
    fig_hist.add_vline(x=r95, line_color='green',     line_dash='dash',  annotation_text=f'95th: {r95:.1f}%')
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- Metrics (show conditional if available, else unconditional) ---
    active = cond_final_ret if has_conditional else unc_final_ret
    fp5, fp50, fp95 = np.percentile(active, [5, 50, 95])
    prob_positive  = float((active > 0).mean() * 100)
    expected_return = float(np.mean(active))
    label = f"after {trigger} triggers" if has_conditional else "unconditional"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${current_price:.2f}",
            help="The last closing price used as the starting point for all simulated paths.")
    with col2:
        st.metric("Expected Return", f"{expected_return:.1f}%",
            f"${current_price * (1 + expected_return / 100):.2f}",
            help=f"Average return across {n_sims_int:,} simulated paths ({label}) at day {sim_days_int}.")
    with col3:
        st.metric("Probability of Gain", f"{prob_positive:.1f}%",
            help=f"Share of {n_sims_int:,} simulations ({label}) that ended above the current price.")
    with col4:
        downside = abs(fp5)
        st.metric("95% Downside Risk", f"{downside:.1f}%",
            f"${current_price * (1 - downside / 100):.2f}",
            delta_color="inverse",
            help=f"Only the worst 5% of simulations ({label}) fell below this price. Stress-test floor, not a guaranteed stop.")

    # --- Comparison table (unconditional vs conditional) ---
    if has_conditional:
        unc_r5, unc_r50, unc_r95 = np.percentile(unc_final_ret, [5, 50, 95])
        unc_prob = float((unc_final_ret > 0).mean() * 100)
        unc_exp  = float(np.mean(unc_final_ret))
        rows = {
            'Metric': ['Expected Return', 'Prob. of Gain', '5th Pct (downside)', 'Median', '95th Pct (upside)'],
            f'Any Day (unconditional)': [
                f"{unc_exp:.1f}%", f"{unc_prob:.1f}%",
                f"{unc_r5:.1f}%", f"{unc_r50:.1f}%", f"{unc_r95:.1f}%"
            ],
            f'After {trigger} Triggers': [
                f"{expected_return:.1f}%", f"{prob_positive:.1f}%",
                f"{fp5:.1f}%", f"{fp50:.1f}%", f"{fp95:.1f}%"
            ],
        }
        with st.expander("📊 Unconditional vs Trigger-Conditioned Comparison"):
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_returns = np.percentile(active, percentiles)
    pct_prices  = current_price * (1 + pct_returns / 100)
    with st.expander("📋 Full Percentile Breakdown"):
        st.dataframe(
            pd.DataFrame({
                'Percentile': [f"{p}th" for p in percentiles],
                'Return': [f"{r:.2f}%" for r in pct_returns],
                'Price': [f"${p:.2f}" for p in pct_prices]
            }),
            use_container_width=True, hide_index=True
        )


def create_benchmark_comparison(result, experiment):
    """Create benchmark comparison for the pattern"""
    st.subheader("🎯 Benchmark Comparison")

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
                st.metric(f"Strategy Return ({best_period})", f"{best_strategy_return:.2f}%")
            with col3:
                outperformance = best_strategy_return - buy_hold_return
                st.metric("Outperformance", f"{outperformance:.2f}%", delta=f"{outperformance:.2f}%")

    except Exception as e:
        st.caption(f"Benchmark comparison unavailable: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("Please check your environment setup and try again.")
