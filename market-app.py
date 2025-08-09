import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.title("🔍 Market Scout - Debug Mode")
st.warning("🐛 DEBUG MODE: This version shows detailed calculations to find the 100% win rate bug")

def debug_forward_returns(target_data, signal_date, forward_periods):
    """Debug version of forward returns calculation"""
    try:
        signal_idx = target_data.index.get_loc(signal_date)
        signal_price = float(target_data.iloc[signal_idx]['Close'])
        
        st.write(f"**Debug Signal Date: {signal_date}**")
        st.write(f"- Signal Index: {signal_idx}")
        st.write(f"- Signal Price: ${signal_price:.2f}")
        st.write(f"- Target Data Length: {len(target_data)}")
        
        returns = {}
        
        for period in forward_periods:
            target_idx = signal_idx + period
            
            st.write(f"\n**{period}-day forward calculation:**")
            st.write(f"- Target Index: {signal_idx} + {period} = {target_idx}")
            
            if target_idx < len(target_data):
                future_date = target_data.index[target_idx]
                future_price = float(target_data.iloc[target_idx]['Close'])
                forward_return = ((future_price - signal_price) / signal_price) * 100
                
                st.write(f"- Future Date: {future_date}")
                st.write(f"- Future Price: ${future_price:.2f}")
                st.write(f"- Return Calculation: ({future_price:.2f} - {signal_price:.2f}) / {signal_price:.2f} * 100")
                st.write(f"- **RESULT: {forward_return:.2f}%**")
                
                returns[f'{period}d'] = forward_return
            else:
                st.write(f"- ❌ Target index {target_idx} >= data length {len(target_data)}")
                returns[f'{period}d'] = None
                
        return returns
        
    except Exception as e:
        st.error(f"Error in debug calculation: {str(e)}")
        return {}

def get_price_data_simple(symbol, start_date, end_date):
    """Simplified data fetching for debugging"""
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            return None, f"No data for {symbol}"
            
        # Handle multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        return data, None
    except Exception as e:
        return None, str(e)

# Simple test interface
st.header("🧪 Quick Debug Test")

col1, col2 = st.columns(2)
with col1:
    trigger_symbol = st.text_input("Trigger Symbol", value="GLD")
    target_symbol = st.text_input("Target Symbol", value="GDX")
    
with col2:
    test_threshold = st.number_input("Price Change %", value=2.0)
    days_back = st.number_input("Days Back", value=90, min_value=30, max_value=365)

end_date = datetime.now().date()
start_date = end_date - timedelta(days=days_back)

if st.button("🔍 Run Debug Test"):
    st.header("📊 Debug Results")
    
    # Fetch data
    st.subheader("1. Data Fetching")
    trigger_data, trigger_error = get_price_data_simple(trigger_symbol, start_date, end_date)
    target_data, target_error = get_price_data_simple(target_symbol, start_date, end_date)
    
    if trigger_error or target_error:
        st.error(f"Data errors: {trigger_error}, {target_error}")
        st.stop()
        
    st.success(f"✅ Fetched {len(trigger_data)} trigger rows, {len(target_data)} target rows")
    
    # Show data samples
    st.subheader("2. Data Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**{trigger_symbol} (Trigger) - Last 5 days:**")
        st.dataframe(trigger_data[['Close']].tail())
    with col2:
        st.write(f"**{target_symbol} (Target) - Last 5 days:**")
        st.dataframe(target_data[['Close']].tail())
    
    # Find signals manually
    st.subheader("3. Signal Detection Debug")
    
    signals_found = []
    
    for i in range(1, min(len(trigger_data), 20)):  # Check first 20 days only for debugging
        try:
            # Simple price change calculation
            prev_price = float(trigger_data.iloc[i-1]['Close'])
            curr_price = float(trigger_data.iloc[i]['Close'])
            pct_change = ((curr_price - prev_price) / prev_price) * 100
            
            signal_date = trigger_data.index[i]
            
            st.write(f"**Day {i}: {signal_date.strftime('%Y-%m-%d')}**")
            st.write(f"- Price: ${prev_price:.2f} → ${curr_price:.2f}")
            st.write(f"- Change: {pct_change:.2f}%")
            
            if pct_change >= test_threshold:
                st.write(f"🎯 **SIGNAL TRIGGERED!** (+{pct_change:.2f}% >= +{test_threshold}%)")
                
                # Check if signal date exists in target data
                if signal_date in target_data.index:
                    st.write("✅ Signal date found in target data")
                    
                    # Calculate forward returns with detailed debug
                    forward_returns = debug_forward_returns(target_data, signal_date, [7, 30])
                    
                    signals_found.append({
                        'date': signal_date,
                        'trigger_change': pct_change,
                        'returns': forward_returns
                    })
                    
                    st.divider()
                else:
                    st.write("❌ Signal date NOT found in target data")
            else:
                st.write("⭕ No signal (below threshold)")
                
        except Exception as e:
            st.write(f"❌ Error on day {i}: {str(e)}")
            
        # Only show first few for debugging
        if len(signals_found) >= 3:
            st.write("... (stopping at 3 signals for debug)")
            break
    
    # Summary
    st.subheader("4. Debug Summary")
    
    if signals_found:
        st.write(f"**Found {len(signals_found)} signals in first 20 days**")
        
        all_7d_returns = []
        all_30d_returns = []
        
        for signal in signals_found:
            if '7d' in signal['returns'] and signal['returns']['7d'] is not None:
                all_7d_returns.append(signal['returns']['7d'])
            if '30d' in signal['returns'] and signal['returns']['30d'] is not None:
                all_30d_returns.append(signal['returns']['30d'])
        
        if all_7d_returns:
            win_rate_7d = (np.array(all_7d_returns) > 0).mean() * 100
            avg_return_7d = np.mean(all_7d_returns)
            st.write(f"**7-day results:** {win_rate_7d:.1f}% win rate, {avg_return_7d:.2f}% avg return")
            st.write(f"Returns: {[f'{r:.2f}%' for r in all_7d_returns]}")
            
        if all_30d_returns:
            win_rate_30d = (np.array(all_30d_returns) > 0).mean() * 100
            avg_return_30d = np.mean(all_30d_returns)
            st.write(f"**30-day results:** {win_rate_30d:.1f}% win rate, {avg_return_30d:.2f}% avg return")
            st.write(f"Returns: {[f'{r:.2f}%' for r in all_30d_returns]}")
            
        # RED FLAGS
        if all_7d_returns and all(r > 0 for r in all_7d_returns):
            st.error("🚨 RED FLAG: ALL 7-day returns are positive!")
        if all_30d_returns and all(r > 0 for r in all_30d_returns):
            st.error("🚨 RED FLAG: ALL 30-day returns are positive!")
            
    else:
        st.warning("No signals found in the test period")
    
    # Market context
    st.subheader("5. Market Context Check")
    
    # Check if we're in a bull market
    start_spy = target_data.iloc[0]['Close'] if 'SPY' in target_symbol.upper() else None
    end_spy = target_data.iloc[-1]['Close'] if 'SPY' in target_symbol.upper() else None
    
    if start_spy and end_spy:
        market_return = ((end_spy - start_spy) / start_spy) * 100
        st.write(f"**Market Return ({target_symbol}):** {market_return:.2f}% over {days_back} days")
        
        if market_return > 10:
            st.warning(f"⚠️ Strong bull market period! (+{market_return:.1f}%) Any strategy might look good.")
    
    # Check individual signal dates
    st.subheader("6. Manual Verification")
    st.write("**Try manually checking one signal:**")
    
    if signals_found:
        first_signal = signals_found[0]
        signal_date = first_signal['date']
        
        st.write(f"Signal Date: {signal_date}")
        st.write(f"Go to Yahoo Finance and check:")
        st.write(f"1. {target_symbol} price on {signal_date}")
        st.write(f"2. {target_symbol} price 7 days later")
        st.write(f"3. Calculate the return manually")
        st.write(f"Our calculation shows: {first_signal['returns'].get('7d', 'N/A')}%")

# Tips section
st.subheader("🔧 Common Issues Causing 100% Win Rates")

st.write("""
**Most Likely Bugs:**

1. **🐛 Forward vs Backward**: Calculating backward returns instead of forward
2. **🐛 Index Error**: Using wrong date indexing (calendar days vs trading days)  
3. **🐛 Bull Market Bias**: Testing only during 2024-2025 mega bull run
4. **🐛 Data Alignment**: Signal dates not matching target data properly
5. **🐛 Price Data Issues**: Using adjusted vs unadjusted prices inconsistently

**Quick Tests:**
- Try testing during 2022 (bear market year)
- Use shorter periods (7 days max)
- Manually verify 2-3 calculations on Yahoo Finance
- Test with SPY → SPY (should be ~0% return)
""")
