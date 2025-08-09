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

st.title("🔍 Market Scout")

# CRITICAL FIX: Clear session state if we detect corruption
if st.button("🔄 Clear Session & Reset"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("Session cleared! Please refresh the page.")
    st.stop()

class MarketScout:
    def __init__(self):
        self.results = []

    def calculate_forward_returns_fixed(self, target_data, signal_date, forward_periods):
        """FIXED: More robust forward returns calculation with detailed logging"""
        try:
            # Get signal index
            if signal_date not in target_data.index:
                return {}
                
            signal_idx = target_data.index.get_loc(signal_date)
            signal_price = float(target_data.iloc[signal_idx]['Close'])

            returns = {}
            
            # CRITICAL FIX: Add debug logging for first few signals
            debug_this_signal = signal_idx < 3
            
            if debug_this_signal:
                st.write(f"🔍 **Debug Signal {signal_idx}:** {signal_date}")
                st.write(f"   Signal Price: ${signal_price:.2f}")

            for period in forward_periods:
                target_idx = signal_idx + period
                
                if target_idx < len(target_data):
                    future_price = float(target_data.iloc[target_idx]['Close'])
                    forward_return = ((future_price - signal_price) / signal_price) * 100
                    returns[f'{period}d'] = forward_return
                    
                    if debug_this_signal:
                        future_date = target_data.index[target_idx]
                        st.write(f"   {period}d: ${signal_price:.2f} → ${future_price:.2f} = **{forward_return:.2f}%** ({future_date.strftime('%m/%d')})")
                else:
                    returns[f'{period}d'] = None
                    if debug_this_signal:
                        st.write(f"   {period}d: ❌ No data (index {target_idx} >= {len(target_data)})")

            return returns

        except Exception as e:
            st.error(f"Forward return calculation error: {str(e)}")
            return {}

    def run_experiment_fixed(self, experiment):
        """FIXED: More robust experiment execution with better error handling"""
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Fetch data
            status_text.text("📈 Fetching data...")
            progress_bar.progress(25)

            trigger_data, trigger_error = self.get_price_data_simple(
                experiment['trigger_symbol'],
                experiment['start_date'],
                experiment['end_date']
            )

            if trigger_error:
                return self._empty_result(experiment['name'], trigger_error)

            # CRITICAL FIX: Extend end date more conservatively
            max_forward_days = max(experiment['forward_periods'])
            extended_end = pd.to_datetime(experiment['end_date']) + timedelta(days=max_forward_days + 30)

            target_data, target_error = self.get_price_data_simple(
                experiment['target_symbol'],
                experiment['start_date'],
                extended_end.strftime('%Y-%m-%d')
            )

            if target_error:
                return self._empty_result(experiment['name'], target_error)

            status_text.text("🔍 Scanning for signals...")
            progress_bar.progress(50)

            signals = []
            total_signals_found = 0
            signals_with_valid_returns = 0

            # CRITICAL FIX: Scan through data more carefully
            for i in range(1, len(trigger_data)):  # Start from 1, not 0
                try:
                    condition_met = self.check_condition_simple(
                        trigger_data,
                        experiment['trigger_condition'],
                        i
                    )

                    if condition_met:
                        total_signals_found += 1
                        signal_date = trigger_data.index[i]

                        # CRITICAL FIX: Ensure signal date exists in target data
                        if signal_date in target_data.index:
                            forward_returns = self.calculate_forward_returns_fixed(
                                target_data,
                                signal_date,
                                experiment['forward_periods']
                            )

                            # CRITICAL FIX: Only add if we have at least one valid return
                            valid_returns = [v for v in forward_returns.values() if v is not None]
                            
                            if valid_returns:
                                signals_with_valid_returns += 1
                                
                                trigger_price = float(trigger_data.iloc[i]['Close'])
                                target_price = float(target_data.loc[signal_date]['Close'])

                                signal_data = {
                                    'date': signal_date.strftime('%Y-%m-%d'),
                                    'trigger_price': trigger_price,
                                    'target_price': target_price,
                                    **forward_returns
                                }

                                signals.append(signal_data)

                except Exception as e:
                    continue

            status_text.text("📊 Calculating metrics...")
            progress_bar.progress(75)

            # Show signal summary
            st.info(f"📊 Found {total_signals_found} total signals, {signals_with_valid_returns} with valid forward data")

            # CRITICAL FIX: Calculate statistics more carefully
            results = {
                'experiment': experiment['name'],
                'trigger_symbol': experiment['trigger_symbol'],
                'target_symbol': experiment['target_symbol'],
                'signals': len(signals),
                'signal_details': signals,
                'forward_analysis': {}
            }

            # CRITICAL FIX: Process each period independently with validation
            for period in experiment['forward_periods']:
                period_key = f'{period}d'
                
                # Extract returns for this specific period
                period_returns = []
                for signal in signals:
                    if period_key in signal and signal[period_key] is not None:
                        period_returns.append(signal[period_key])

                if period_returns:
                    period_returns_array = np.array(period_returns)
                    
                    # CRITICAL FIX: Add validation checks
                    positive_returns = period_returns_array > 0
                    negative_returns = period_returns_array < 0
                    zero_returns = period_returns_array == 0
                    
                    win_rate = float(positive_returns.mean() * 100)
                    
                    # CRITICAL FIX: Warn about suspicious results
                    if win_rate > 95 and len(period_returns) > 5:
                        st.warning(f"⚠️ Suspicious {period_key} win rate: {win_rate:.1f}% - This may indicate a bug!")
                        st.write(f"Returns breakdown: {len(period_returns)} total, {positive_returns.sum()} wins, {negative_returns.sum()} losses")
                    
                    results['forward_analysis'][period_key] = {
                        'valid_signals': len(period_returns),
                        'avg_return': float(np.mean(period_returns_array)),
                        'win_rate': win_rate,
                        'best_signal': float(np.max(period_returns_array)),
                        'worst_signal': float(np.min(period_returns_array)),
                        'std_dev': float(np.std(period_returns_array)),
                        'total_return': float(np.sum(period_returns_array)),
                        'wins': int(positive_returns.sum()),
                        'losses': int(negative_returns.sum()),
                        'zeros': int(zero_returns.sum())
                    }
                else:
                    results['forward_analysis'][period_key] = {
                        'valid_signals': 0,
                        'avg_return': 0,
                        'win_rate': 0,
                        'best_signal': 0,
                        'worst_signal': 0,
                        'std_dev': 0,
                        'total_return': 0,
                        'wins': 0,
                        'losses': 0,
                        'zeros': 0
                    }

            status_text.text("✅ Complete!")
            progress_bar.progress(100)

            # Clear progress indicators
            import time
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()

            return results

        except Exception as e:
            st.error(f"Experiment failed: {str(e)}")
            return self._empty_result(experiment['name'], str(e))

    def check_condition_simple(self, data, condition, date_idx):
        """Simplified condition checking"""
        try:
            if date_idx < condition.get('lookback_days', 1):
                return False

            if condition['type'] == 'price_change':
                lookback = condition.get('lookback_days', 1)
                threshold = float(condition['threshold_pct'])
                direction = condition['direction']

                start_price = float(data['Close'].iloc[date_idx - lookback])
                end_price = float(data['Close'].iloc[date_idx])
                pct_change = ((end_price - start_price) / start_price) * 100

                if direction == 'up':
                    return pct_change >= threshold
                else:
                    return pct_change <= -threshold

            return False

        except Exception as e:
            return False

    def get_price_data_simple(self, symbol, start_date, end_date):
        """Simplified data fetching"""
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                return None, f"No data for {symbol}"
                
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                
            return data, None
        except Exception as e:
            return None, str(e)

    def _empty_result(self, name, error):
        return {
            'experiment': name,
            'signals': 0,
            'forward_analysis': {},
            'signal_details': [],
            'error': error
        }

# Initialize session state more carefully
if 'experiments' not in st.session_state:
    st.session_state.experiments = []

if 'scout' not in st.session_state:
    st.session_state.scout = MarketScout()

# Simple interface for quick testing
st.header("🧪 Quick Test")

col1, col2 = st.columns(2)
with col1:
    trigger_symbol = st.text_input("Trigger Symbol", value="GLD")
    target_symbol = st.text_input("Target Symbol", value="GDX") 
    
with col2:
    threshold = st.number_input("Threshold %", value=2.0)
    days_back = st.number_input("Days Back", value=180, min_value=90, max_value=730)

# CRITICAL FIX: Test different time periods
st.write("**Test different periods to check for bull market bias:**")
test_period = st.selectbox("Test Period", [
    "Recent (last 180 days)",
    "2024 Full Year", 
    "2023 Full Year",
    "2022 Bear Market",
    "2021-2022 Mixed"
])

if test_period == "Recent (last 180 days)":
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
elif test_period == "2024 Full Year":
    start_date = datetime(2024, 1, 1).date()
    end_date = datetime(2024, 12, 31).date()
elif test_period == "2023 Full Year":
    start_date = datetime(2023, 1, 1).date()
    end_date = datetime(2023, 12, 31).date()
elif test_period == "2022 Bear Market":
    start_date = datetime(2022, 1, 1).date()
    end_date = datetime(2022, 12, 31).date()
else:  # 2021-2022 Mixed
    start_date = datetime(2021, 6, 1).date()
    end_date = datetime(2022, 12, 31).date()

if st.button("🚀 Run Fixed Test", type="primary"):
    experiment = {
        'name': f"{trigger_symbol} → {target_symbol} ({test_period})",
        'trigger_symbol': trigger_symbol,
        'target_symbol': target_symbol,
        'trigger_condition': {
            'type': 'price_change',
            'direction': 'up',
            'threshold_pct': threshold,
            'lookback_days': 1
        },
        'forward_periods': [7, 30],
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d')
    }
    
    st.header("📊 Fixed Results")
    result = st.session_state.scout.run_experiment_fixed(experiment)
    
    if result.get('error'):
        st.error(f"❌ {result['error']}")
    elif result['signals'] == 0:
        st.warning("⚠️ No signals found")
    else:
        st.success(f"✅ Found {result['signals']} signals")
        
        # Show detailed results
        forward_analysis = result.get('forward_analysis', {})
        
        for period, stats in forward_analysis.items():
            if stats['valid_signals'] > 0:
                st.subheader(f"📈 {period} Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Signals", stats['valid_signals'])
                with col2:
                    st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
                with col3:
                    st.metric("Avg Return", f"{stats['avg_return']:.2f}%")
                with col4:
                    st.metric("W/L Ratio", f"{stats['wins']}/{stats['losses']}")
                
                st.write(f"**Returns Range:** {stats['worst_signal']:.2f}% to {stats['best_signal']:.2f}%")
                
                # Reality check
                if stats['win_rate'] > 90:
                    st.error(f"🚨 {stats['win_rate']:.0f}% win rate is unrealistic! Possible issues:")
                    st.write("- Bull market bias (test during 2022 bear market)")
                    st.write("- Data quality issues")
                    st.write("- Calculation bugs")
                elif stats['win_rate'] > 75:
                    st.warning(f"⚠️ {stats['win_rate']:.0f}% win rate is very high - verify manually")
                else:
                    st.success(f"✅ {stats['win_rate']:.0f}% win rate looks realistic")

# Show sample calculations
st.subheader("💡 Tips to Verify Results")
st.write("""
**Manual Verification Steps:**
1. **Pick one signal date** from the results above
2. **Check Yahoo Finance** for exact prices on that date and X days later  
3. **Calculate return manually**: (Future Price - Signal Price) / Signal Price × 100
4. **Compare** with our calculated result

**Red Flags:**
- Win rates > 90% (almost impossible)
- All returns positive during any 90+ day period
- Results too good to be true (usually are!)

**Good Signs:**  
- Win rates 55-75% (realistic for good strategies)
- Mix of positive and negative returns
- Results worse during bear markets (2022)
""")
