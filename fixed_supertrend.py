#!/usr/bin/env python3
"""
Fixed SuperTrend ATR Buffer Strategy Implementation
Corrected to match TradingView Pine Script behavior exactly

Author: arullr001
Date: 2025-01-27
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import traceback

warnings.filterwarnings('ignore')

def calculate_true_range(high, low, close):
    """
    Calculate True Range using TradingView's standard method
    """
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    
    # True Range calculation
    tr1 = high - low  # High - Low
    tr2 = np.abs(high - np.roll(close, 1))  # High - Previous Close
    tr3 = np.abs(low - np.roll(close, 1))   # Low - Previous Close
    
    # For the first bar, use High - Low
    tr2[0] = 0
    tr3[0] = 0
    
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    return true_range

def calculate_atr_rma(true_range, period):
    """
    Calculate ATR using RMA (Running Moving Average) exactly like TradingView
    RMA is different from SMA - it's an exponentially weighted average
    """
    atr = np.zeros_like(true_range)
    
    # First value is just the true range
    atr[0] = true_range[0]
    
    # RMA calculation: atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    for i in range(1, len(true_range)):
        atr[i] = (atr[i-1] * (period - 1) + true_range[i]) / period
    
    return atr

def calculate_supertrend_tradingview(df, atr_period=10, factor=3.0, buffer_multiplier=0.3):
    """
    Calculate SuperTrend exactly as TradingView Pine Script does
    
    This implementation matches TradingView's SuperTrend indicator precisely:
    - Uses RMA for ATR calculation (not SMA)
    - Correct direction values (1 = uptrend, -1 = downtrend)
    - Proper upper/lower band calculation
    - Buffer zones for entry signals
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: high, low, close (and optionally open)
    atr_period : int
        Period for ATR calculation (default: 10)
    factor : float  
        SuperTrend factor (default: 3.0)
    buffer_multiplier : float
        Buffer zone distance as multiple of ATR (default: 0.3)
        
    Returns:
    --------
    pandas.DataFrame with additional columns:
        - atr: Average True Range
        - basic_ub: Basic upper band
        - basic_lb: Basic lower band  
        - final_ub: Final upper band
        - final_lb: Final lower band
        - supertrend: SuperTrend line
        - direction: Trend direction (1=up, -1=down)
        - up_trend_buffer: Upper trend buffer zone
        - down_trend_buffer: Lower trend buffer zone
        - buy_signal: Long entry signal
        - sell_signal: Short entry signal
    """
    print(f"Calculating SuperTrend: ATR={atr_period}, Factor={factor}, Buffer={buffer_multiplier}")
    
    # Validate input
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # Work with a copy
    result_df = df.copy()
    n = len(df)
    
    # Extract price arrays
    high = df['high'].values
    low = df['low'].values  
    close = df['close'].values
    
    # Calculate True Range and ATR using TradingView method
    tr = calculate_true_range(high, low, close)
    atr = calculate_atr_rma(tr, atr_period)
    
    # Calculate HL2 (median price)
    hl2 = (high + low) / 2
    
    # Calculate basic upper and lower bands
    basic_ub = hl2 + (factor * atr)
    basic_lb = hl2 - (factor * atr)
    
    # Initialize final bands
    final_ub = np.zeros(n)
    final_lb = np.zeros(n)
    
    # Set initial values
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    
    # Calculate final bands using TradingView logic
    for i in range(1, n):
        # Upper band calculation
        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        # Lower band calculation  
        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # Calculate SuperTrend and direction
    supertrend = np.zeros(n)
    direction = np.zeros(n, dtype=int)
    
    # Initialize first values
    supertrend[0] = final_ub[0]
    direction[0] = 1  # Start with uptrend
    
    # Calculate SuperTrend values
    for i in range(1, n):
        # Determine current trend based on close vs previous supertrend
        if close[i] > final_ub[i]:
            direction[i] = 1  # Uptrend
            supertrend[i] = final_lb[i]
        elif close[i] < final_lb[i]:
            direction[i] = -1  # Downtrend  
            supertrend[i] = final_ub[i]
        else:
            # Price is between bands, continue previous trend
            direction[i] = direction[i-1]
            if direction[i] == 1:
                supertrend[i] = final_lb[i]
            else:
                supertrend[i] = final_ub[i]
    
    # Calculate buffer zones
    buffer_distance = atr * buffer_multiplier
    
    # Buffer zones are only active in their respective trends
    up_trend_buffer = np.where(direction == 1, supertrend + buffer_distance, np.nan)
    down_trend_buffer = np.where(direction == -1, supertrend - buffer_distance, np.nan)
    
    # Add all calculations to dataframe
    result_df['atr'] = atr
    result_df['basic_ub'] = basic_ub
    result_df['basic_lb'] = basic_lb
    result_df['final_ub'] = final_ub
    result_df['final_lb'] = final_lb
    result_df['supertrend'] = supertrend
    result_df['direction'] = direction
    result_df['up_trend_buffer'] = up_trend_buffer
    result_df['down_trend_buffer'] = down_trend_buffer
    
    # Generate trading signals
    # Long entry: In uptrend AND price touches upper buffer (price >= buffer)
    # Short entry: In downtrend AND price touches lower buffer (price <= buffer)
    
    buy_signal = np.zeros(n, dtype=bool)
    sell_signal = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        # Long entry signal: trend changed to up AND price touching upper buffer
        if (direction[i] == 1 and direction[i-1] == -1 and 
            not np.isnan(up_trend_buffer[i]) and close[i] >= up_trend_buffer[i]):
            buy_signal[i] = True
            
        # Short entry signal: trend changed to down AND price touching lower buffer  
        if (direction[i] == -1 and direction[i-1] == 1 and
            not np.isnan(down_trend_buffer[i]) and close[i] <= down_trend_buffer[i]):
            sell_signal[i] = True
    
    result_df['buy_signal'] = buy_signal
    result_df['sell_signal'] = sell_signal
    
    # Debug output
    total_buy = buy_signal.sum()
    total_sell = sell_signal.sum()
    trend_changes = np.sum(direction[1:] != direction[:-1])
    
    print(f"SuperTrend Calculation Results:")
    print(f"  - Trend changes: {trend_changes}")
    print(f"  - Buy signals: {total_buy}")
    print(f"  - Sell signals: {total_sell}")
    print(f"  - Current trend: {'Up' if direction[-1] == 1 else 'Down'}")
    print(f"  - Last SuperTrend value: {supertrend[-1]:.2f}")
    
    return result_df

def backtest_strategy(df, atr_period=10, factor=3.0, buffer_multiplier=0.3, 
                     hard_stop_distance=50, initial_capital=10000):
    """
    Backtest the corrected SuperTrend ATR Buffer strategy
    
    Parameters:
    -----------
    df : pandas.DataFrame
        OHLC data with SuperTrend calculations
    atr_period : int
        ATR period 
    factor : float
        SuperTrend factor
    buffer_multiplier : float  
        Buffer zone multiplier
    hard_stop_distance : float
        Hard stop distance in points
    initial_capital : float
        Starting capital
        
    Returns:
    --------
    dict: Backtest results with trades and performance metrics
    """
    
    # Calculate SuperTrend with corrected logic
    st_df = calculate_supertrend_tradingview(df, atr_period, factor, buffer_multiplier)
    
    # Initialize tracking variables
    trades = []
    equity = initial_capital
    max_equity = initial_capital
    drawdown_curve = []
    position = 0  # 0 = no position, 1 = long, -1 = short
    entry_price = 0
    entry_time = None
    trade_id = 0
    
    # Convert to numpy for speed
    high_vals = st_df['high'].values
    low_vals = st_df['low'].values
    close_vals = st_df['close'].values
    supertrend_vals = st_df['supertrend'].values
    direction_vals = st_df['direction'].values
    buy_signals = st_df['buy_signal'].values
    sell_signals = st_df['sell_signal'].values
    timestamps = st_df.index
    
    print(f"\nRunning backtest with {len(st_df)} bars...")
    
    for i in range(1, len(st_df)):
        current_price = close_vals[i]
        current_time = timestamps[i]
        
        # Check for position exits first
        if position == 1:  # Long position
            exit_triggered = False
            exit_price = current_price
            exit_reason = "unknown"
            
            # Check hard stop
            if low_vals[i] <= entry_price - hard_stop_distance:
                exit_triggered = True
                exit_price = max(low_vals[i], entry_price - hard_stop_distance)
                exit_reason = "hard_stop"
            # Check trend change
            elif direction_vals[i] == -1:  # Trend changed to down
                exit_triggered = True
                exit_reason = "trend_change"
                
            if exit_triggered:
                # Calculate trade result
                pnl = exit_price - entry_price
                pnl_pct = (pnl / entry_price) * 100
                duration = (current_time - entry_time).total_seconds() / 3600  # hours
                
                trades.append({
                    'trade_id': trade_id,
                    'direction': 'long',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': current_time,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'duration_hours': duration,
                    'exit_reason': exit_reason
                })
                
                # Update equity
                equity += pnl
                max_equity = max(max_equity, equity)
                drawdown = (max_equity - equity) / max_equity * 100
                drawdown_curve.append(drawdown)
                
                position = 0
                print(f"Exit Long #{trade_id}: {exit_price:.2f} ({exit_reason}) - P&L: {pnl:.2f} ({pnl_pct:.2f}%)")
        
        elif position == -1:  # Short position
            exit_triggered = False
            exit_price = current_price
            exit_reason = "unknown"
            
            # Check hard stop
            if high_vals[i] >= entry_price + hard_stop_distance:
                exit_triggered = True
                exit_price = min(high_vals[i], entry_price + hard_stop_distance)
                exit_reason = "hard_stop"
            # Check trend change
            elif direction_vals[i] == 1:  # Trend changed to up
                exit_triggered = True
                exit_reason = "trend_change"
                
            if exit_triggered:
                # Calculate trade result
                pnl = entry_price - exit_price  # Reversed for short
                pnl_pct = (pnl / entry_price) * 100
                duration = (current_time - entry_time).total_seconds() / 3600  # hours
                
                trades.append({
                    'trade_id': trade_id,
                    'direction': 'short',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': current_time,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'duration_hours': duration,
                    'exit_reason': exit_reason
                })
                
                # Update equity
                equity += pnl
                max_equity = max(max_equity, equity)
                drawdown = (max_equity - equity) / max_equity * 100
                drawdown_curve.append(drawdown)
                
                position = 0
                print(f"Exit Short #{trade_id}: {exit_price:.2f} ({exit_reason}) - P&L: {pnl:.2f} ({pnl_pct:.2f}%)")
        
        # Check for new entries (only when no position)
        if position == 0:
            if buy_signals[i]:
                trade_id += 1
                position = 1
                entry_price = current_price
                entry_time = current_time
                print(f"Enter Long #{trade_id}: {entry_price:.2f} at {entry_time}")
                
            elif sell_signals[i]:
                trade_id += 1
                position = -1
                entry_price = current_price
                entry_time = current_time
                print(f"Enter Short #{trade_id}: {entry_price:.2f} at {entry_time}")
    
    # Calculate performance metrics
    if trades:
        profits = [t['pnl'] for t in trades]
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        total_return = (equity - initial_capital) / initial_capital * 100
        win_rate = len(winning_trades) / len(trades) * 100
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades else float('inf')
        max_drawdown = max(drawdown_curve) if drawdown_curve else 0
        avg_duration = np.mean([t['duration_hours'] for t in trades])
        
        # Sharpe ratio (simplified)
        returns = [t['pnl_pct'] for t in trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        performance = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_duration_hours': avg_duration,
            'sharpe_ratio': sharpe_ratio,
            'final_equity': equity
        }
    else:
        performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'avg_duration_hours': 0,
            'sharpe_ratio': 0,
            'final_equity': initial_capital
        }
    
    return {
        'trades': trades,
        'performance': performance,
        'equity_curve': drawdown_curve,
        'supertrend_data': st_df
    }

def load_data(file_path):
    """
    Load OHLC data from CSV file with proper date handling
    """
    print(f"Loading data from: {file_path}")
    
    try:
        # Read CSV
        df = pd.read_csv(file_path)
        print(f"Raw data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Standardize column names
        column_mapping = {
            'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close',
            'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close'
        }
        df = df.rename(columns=column_mapping)
        
        # Handle datetime
        if 'date' in df.columns and 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        elif 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])
        else:
            # Assume index is already datetime or create sequential
            df['datetime'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1T')
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        
        # Keep only OHLC columns
        required_cols = ['open', 'high', 'low', 'close']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) < 4:
            raise ValueError(f"Missing required columns. Available: {available_cols}, Required: {required_cols}")
        
        df = df[available_cols]
        
        # Remove any rows with NaN
        df = df.dropna()
        
        # Ensure data types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()  # Remove any conversion failures
        
        print(f"Processed data shape: {df.shape}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"Sample data:\n{df.head()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        traceback.print_exc()
        return None

def apply_date_filter(df, start_date=None, end_date=None):
    """
    Filter dataframe by date range
    """
    if start_date is None and end_date is None:
        return df
    
    filtered_df = df.copy()
    
    if start_date:
        start_date = pd.to_datetime(start_date)
        filtered_df = filtered_df[filtered_df.index >= start_date]
        
    if end_date:
        end_date = pd.to_datetime(end_date)
        filtered_df = filtered_df[filtered_df.index <= end_date]
    
    print(f"Date filter applied: {len(filtered_df)} rows (from {len(df)} original)")
    if len(filtered_df) > 0:
        print(f"Filtered date range: {filtered_df.index[0]} to {filtered_df.index[-1]}")
    
    return filtered_df

def plot_results(results):
    """
    Create visualization of backtest results
    """
    try:
        st_df = results['supertrend_data']
        trades = results['trades']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Price and SuperTrend
        ax1.plot(st_df.index, st_df['close'], label='Close Price', color='black', linewidth=0.8)
        
        # Color SuperTrend based on direction
        uptrend_mask = st_df['direction'] == 1
        downtrend_mask = st_df['direction'] == -1
        
        ax1.plot(st_df.index[uptrend_mask], st_df['supertrend'][uptrend_mask], 
                color='green', label='SuperTrend (Up)', linewidth=1.2)
        ax1.plot(st_df.index[downtrend_mask], st_df['supertrend'][downtrend_mask], 
                color='red', label='SuperTrend (Down)', linewidth=1.2)
        
        # Plot buffer zones
        ax1.plot(st_df.index, st_df['up_trend_buffer'], color='green', alpha=0.5, 
                linestyle='--', label='Up Buffer', linewidth=0.8)
        ax1.plot(st_df.index, st_df['down_trend_buffer'], color='red', alpha=0.5, 
                linestyle='--', label='Down Buffer', linewidth=0.8)
        
        # Plot trade entry points
        for trade in trades:
            entry_time = trade['entry_time']
            entry_price = trade['entry_price']
            direction = trade['direction']
            
            if direction == 'long':
                ax1.scatter(entry_time, entry_price, color='green', marker='^', s=100, zorder=5)
            else:
                ax1.scatter(entry_time, entry_price, color='red', marker='v', s=100, zorder=5)
        
        ax1.set_title('SuperTrend ATR Buffer Strategy - Price Chart')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Equity curve
        if trades:
            equity_values = []
            equity_times = []
            current_equity = 10000  # initial capital
            
            equity_values.append(current_equity)
            equity_times.append(st_df.index[0])
            
            for trade in trades:
                current_equity += trade['pnl']
                equity_values.append(current_equity)
                equity_times.append(trade['exit_time'])
            
            ax2.plot(equity_times, equity_values, color='blue', linewidth=1.5)
            ax2.set_title('Equity Curve')
            ax2.set_ylabel('Equity')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No trades generated', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Equity Curve - No Trades')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error creating plots: {str(e)}")
        traceback.print_exc()

def print_performance_summary(results):
    """
    Print detailed performance summary
    """
    performance = results['performance']
    trades = results['trades']
    
    print("\n" + "=" * 60)
    print(" BACKTEST PERFORMANCE SUMMARY ".center(60, "="))
    print("=" * 60)
    
    print(f"Total Trades: {performance['total_trades']}")
    print(f"Winning Trades: {performance['winning_trades']}")
    print(f"Losing Trades: {performance['losing_trades']}")
    print(f"Win Rate: {performance['win_rate']:.2f}%")
    print(f"Total Return: {performance['total_return']:.2f}%")
    print(f"Profit Factor: {performance['profit_factor']:.2f}")
    print(f"Max Drawdown: {performance['max_drawdown']:.2f}%")
    print(f"Average Win: {performance['avg_win']:.2f}")
    print(f"Average Loss: {performance['avg_loss']:.2f}")
    print(f"Average Trade Duration: {performance['avg_duration_hours']:.2f} hours")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"Final Equity: {performance['final_equity']:.2f}")
    
    if trades:
        print("\n" + "-" * 60)
        print(" RECENT TRADES ")
        print("-" * 60)
        
        # Show last 10 trades
        recent_trades = trades[-10:]
        for trade in recent_trades:
            direction_symbol = "▲" if trade['direction'] == 'long' else "▼"
            pnl_color = "+" if trade['pnl'] > 0 else ""
            print(f"#{trade['trade_id']:3d} {direction_symbol} {trade['entry_time'].strftime('%m/%d %H:%M')} → "
                  f"{trade['exit_time'].strftime('%m/%d %H:%M')} | "
                  f"{trade['entry_price']:8.2f} → {trade['exit_price']:8.2f} | "
                  f"{pnl_color}{trade['pnl']:7.2f} ({trade['pnl_pct']:+6.2f}%) | "
                  f"{trade['duration_hours']:5.1f}h | {trade['exit_reason']}")
    
    print("=" * 60)

if __name__ == "__main__":
    print("SuperTrend ATR Buffer Strategy - Fixed Implementation")
    print("=" * 60)
    
    # Test with sample data file
    test_file = "/home/runner/work/STBacktest/STBacktest/btcusd_1m_20240401_to_20250606_gen20250607_233731.csv"
    
    if os.path.exists(test_file):
        print(f"Testing with: {os.path.basename(test_file)}")
        
        # Load data
        df = load_data(test_file)
        if df is not None:
            # Test with a smaller sample first (last 10000 bars)
            df_sample = df.tail(10000)
            print(f"Using sample of {len(df_sample)} bars for testing")
            
            # Run backtest with default parameters
            print("\nRunning backtest with default parameters...")
            results = backtest_strategy(
                df_sample,
                atr_period=10,
                factor=3.0,
                buffer_multiplier=0.3,
                hard_stop_distance=50
            )
            
            # Display results
            print_performance_summary(results)
            
            # Create plots
            try:
                plot_results(results)
            except Exception as e:
                print(f"Plotting failed: {str(e)}")
        else:
            print("Failed to load test data")
    else:
        print(f"Test file not found: {test_file}")
        print("Please update the test_file path to point to your OHLC data")