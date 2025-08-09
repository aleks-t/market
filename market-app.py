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

            # Visualization
            st.subheader("📊 Performance by Time Period")

            periods = [data['Period'] for data in summary_data]
            avg_returns = [float(data['Avg Return'].strip('%')) for data in summary_data]
            win_rates = [float(data['Win Rate'].strip('%')) for data in summary_data]

            fig = go.Figure()

            # Add bars and line
            fig.add_trace(go.Bar(
                x=periods,
                y=avg_returns,
                name='Avg Return %',
                marker_color='lightblue',
                yaxis='y'
            ))

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

                # Choose primary period for timeline
                primary_period = None
                for period in ['90d', '60d', '30d', '15d', '7d']:
                    if period in signals_df.columns and signals_df[period].notna().any():
                        primary_period = period
                        break

                if primary_period:
                    # Create cumulative return chart
                    valid_signals = signals_df[signals_df[primary_period].notna()].copy()
                    valid_signals['cumulative_return'] = valid_signals[primary_period].cumsum()
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

            if best_period:
                best_stats = forward_analysis[best_period]

                # Reality check warning for unrealistic win rates
                if best_stats['win_rate'] > 95:
                    st.warning(f"""
                    ⚠️ **Reality Check**: {best_stats['win_rate']:.0f}% win rate is extremely suspicious!
                    
                    **Why this is likely misleading:**
                    • Testing during bull market period
                    • Long holding periods mask true signal strength
                    • Any random strategy would look good in strong markets
                    
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
                    # Calculate annualized return estimate
                    period_days = len(pd.date_range(experiment['start_date'], experiment['end_date']))
                    annual_return = (best_stats['avg_return'] * best_stats['valid_signals'] * 365) / period_days
                    st.metric(
                        "Annualized Return", 
                        f"{annual_return:.1f}%",
                        help="📈 Expected yearly return if you followed this strategy consistently. Compare to S&P 500 (~10% annually). Above 20% is very good."
                    )

                # Risk metrics
                st.subheader("📊 Risk Metrics")
                col1, col2, col3 = st.columns(3)

                with col1:
                    sharpe_approx = (best_stats['avg_return'] / best_stats['std_dev']) if best_stats['std_dev'] > 0 else 0
                    st.metric(
                        "Sharpe Ratio (approx)", 
                        f"{sharpe_approx:.2f}",
                        help="Risk-adjusted return measure. >0.5 is decent, >1.0 is great, >2.0 is excellent. Higher = better returns for the risk taken."
                    )

                with col2:
                    max_loss = abs(best_stats['worst_signal'])
                    st.metric(
                        "Max Single Loss", 
                        f"{max_loss:.2f}%",
                        help="The worst single trade result - your maximum possible loss on any one trade. This is your worst-case scenario."
                    )

                with col3:
                    profit_factor = abs(best_stats['best_signal'] / best_stats['worst_signal']) if best_stats['worst_signal'] != 0 else float('inf')
                    st.metric(
                        "Best/Worst Ratio", 
                        f"{profit_factor:.2f}",
                        help="Best trade ÷ Worst trade. Shows if your wins are bigger than your losses. >2.0 means wins are twice as big as losses."
                    )

            # AI Analysis
            if experiment.get('enable_ai', True):
                with st.expander("🎓 AI Analysis & Trading Strategy", expanded=True):
                    with st.spinner("Generating AI analysis..."):
                        analysis = get_ai_analysis(
                            result['experiment'],
                            experiment['trigger_symbol'],
                            experiment['target_symbol'],
                            result
                        )
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
