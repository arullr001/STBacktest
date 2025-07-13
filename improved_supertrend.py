#!/usr/bin/env python3
"""
Improved SuperTrend ATR Buffer Strategy Implementation
Fixed to match TradingView Pine Script behavior with enhanced features

Features:
- Corrected SuperTrend calculation matching TradingView exactly
- Proper buffer zone logic for entry signals  
- ATR-based hard stops (instead of fixed points)
- Date range filtering
- Parameter optimization suggestions
- Enhanced debugging and validation

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
import json

warnings.filterwarnings('ignore')

class SuperTrendStrategy:
    """
    SuperTrend ATR Buffer Strategy implementation
    """
    
    def __init__(self, atr_period=10, factor=3.0, buffer_multiplier=0.3, 
                 hard_stop_atr_multiple=2.0, time_exit_hours=24):
        """
        Initialize strategy parameters
        
        Parameters:
        -----------
        atr_period : int
            Period for ATR calculation (default: 10)
        factor : float  
            SuperTrend factor (default: 3.0)
        buffer_multiplier : float
            Buffer zone distance as multiple of ATR (default: 0.3)
        hard_stop_atr_multiple : float
            Hard stop distance as multiple of ATR (default: 2.0)
        time_exit_hours : int
            Maximum hours to hold a trade (default: 24, 0 = disabled)
        """
        self.atr_period = atr_period
        self.factor = factor
        self.buffer_multiplier = buffer_multiplier
        self.hard_stop_atr_multiple = hard_stop_atr_multiple
        self.time_exit_hours = time_exit_hours
        
    def calculate_true_range(self, high, low, close):
        """Calculate True Range using TradingView's standard method"""
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

    def calculate_atr_rma(self, true_range, period):
        """Calculate ATR using RMA (Running Moving Average) exactly like TradingView"""
        atr = np.zeros_like(true_range)
        
        # First value is just the true range
        atr[0] = true_range[0]
        
        # RMA calculation: atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        for i in range(1, len(true_range)):
            atr[i] = (atr[i-1] * (period - 1) + true_range[i]) / period
        
        return atr

    def calculate_supertrend(self, df):
        """
        Calculate SuperTrend exactly as TradingView Pine Script does
        
        Returns DataFrame with additional columns:
        - atr: Average True Range
        - supertrend: SuperTrend line
        - direction: Trend direction (1=up, -1=down)
        - up_trend_buffer: Upper trend buffer zone
        - down_trend_buffer: Lower trend buffer zone
        - buy_signal: Long entry signal
        - sell_signal: Short entry signal
        """
        print(f"Calculating SuperTrend: ATR={self.atr_period}, Factor={self.factor}, Buffer={self.buffer_multiplier}")
        
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
        tr = self.calculate_true_range(high, low, close)
        atr = self.calculate_atr_rma(tr, self.atr_period)
        
        # Calculate HL2 (median price)
        hl2 = (high + low) / 2
        
        # Calculate basic upper and lower bands
        basic_ub = hl2 + (self.factor * atr)
        basic_lb = hl2 - (self.factor * atr)
        
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
        buffer_distance = atr * self.buffer_multiplier
        
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
        
        # Generate trading signals with improved logic
        buy_signal = np.zeros(n, dtype=bool)
        sell_signal = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            # Long entry signal: trend just changed to up AND price touches upper buffer
            if (direction[i] == 1 and direction[i-1] == -1):
                # Check if price is touching the buffer zone (close is at or above buffer)
                if not np.isnan(up_trend_buffer[i]) and close[i] >= up_trend_buffer[i]:
                    buy_signal[i] = True
                    
            # Short entry signal: trend just changed to down AND price touches lower buffer  
            if (direction[i] == -1 and direction[i-1] == 1):
                # Check if price is touching the buffer zone (close is at or below buffer)
                if not np.isnan(down_trend_buffer[i]) and close[i] <= down_trend_buffer[i]:
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
        print(f"  - Average ATR: {np.mean(atr):.2f}")
        
        return result_df

    def backtest(self, df, start_date=None, end_date=None, initial_capital=10000):
        """
        Backtest the strategy with improved risk management
        
        Parameters:
        -----------
        df : pandas.DataFrame
            OHLC data 
        start_date : str or datetime
            Start date for backtest (format: 'YYYY-MM-DD' or datetime object)
        end_date : str or datetime  
            End date for backtest (format: 'YYYY-MM-DD' or datetime object)
        initial_capital : float
            Starting capital
            
        Returns:
        --------
        dict: Backtest results with trades and performance metrics
        """
        
        # Apply date filtering if specified
        filtered_df = self._apply_date_filter(df, start_date, end_date)
        
        if len(filtered_df) == 0:
            raise ValueError("No data available for the specified date range")
        
        # Calculate SuperTrend with corrected logic
        st_df = self.calculate_supertrend(filtered_df)
        
        # Initialize tracking variables
        trades = []
        equity = initial_capital
        max_equity = initial_capital
        drawdown_curve = []
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        entry_time = None
        trade_id = 0
        hard_stop_price = 0
        
        # Convert to numpy for speed
        high_vals = st_df['high'].values
        low_vals = st_df['low'].values
        close_vals = st_df['close'].values
        supertrend_vals = st_df['supertrend'].values
        direction_vals = st_df['direction'].values
        buy_signals = st_df['buy_signal'].values
        sell_signals = st_df['sell_signal'].values
        atr_vals = st_df['atr'].values
        timestamps = st_df.index
        
        print(f"\nRunning backtest with {len(st_df)} bars...")
        print(f"Date range: {timestamps[0]} to {timestamps[-1]}")
        
        for i in range(1, len(st_df)):
            current_price = close_vals[i]
            current_time = timestamps[i]
            current_atr = atr_vals[i]
            
            # Check for position exits first
            if position == 1:  # Long position
                exit_triggered = False
                exit_price = current_price
                exit_reason = "unknown"
                
                # Check hard stop (ATR-based)
                if low_vals[i] <= hard_stop_price:
                    exit_triggered = True
                    exit_price = max(low_vals[i], hard_stop_price)
                    exit_reason = "hard_stop"
                # Check trend change
                elif direction_vals[i] == -1:  # Trend changed to down
                    exit_triggered = True
                    exit_reason = "trend_change"
                # Check time exit
                elif self.time_exit_hours > 0:
                    hours_in_trade = (current_time - entry_time).total_seconds() / 3600
                    if hours_in_trade >= self.time_exit_hours:
                        exit_triggered = True
                        exit_reason = "time_exit"
                        
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
                        'exit_reason': exit_reason,
                        'hard_stop_price': hard_stop_price
                    })
                    
                    # Update equity
                    equity += pnl
                    max_equity = max(max_equity, equity)
                    drawdown = (max_equity - equity) / max_equity * 100
                    drawdown_curve.append(drawdown)
                    
                    position = 0
                    if abs(pnl) > current_atr * 0.5:  # Only log significant trades
                        print(f"Exit Long #{trade_id}: {exit_price:.2f} ({exit_reason}) - P&L: {pnl:.2f} ({pnl_pct:.2f}%)")
            
            elif position == -1:  # Short position
                exit_triggered = False
                exit_price = current_price
                exit_reason = "unknown"
                
                # Check hard stop (ATR-based)
                if high_vals[i] >= hard_stop_price:
                    exit_triggered = True
                    exit_price = min(high_vals[i], hard_stop_price)
                    exit_reason = "hard_stop"
                # Check trend change
                elif direction_vals[i] == 1:  # Trend changed to up
                    exit_triggered = True
                    exit_reason = "trend_change"
                # Check time exit
                elif self.time_exit_hours > 0:
                    hours_in_trade = (current_time - entry_time).total_seconds() / 3600
                    if hours_in_trade >= self.time_exit_hours:
                        exit_triggered = True
                        exit_reason = "time_exit"
                        
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
                        'exit_reason': exit_reason,
                        'hard_stop_price': hard_stop_price
                    })
                    
                    # Update equity
                    equity += pnl
                    max_equity = max(max_equity, equity)
                    drawdown = (max_equity - equity) / max_equity * 100
                    drawdown_curve.append(drawdown)
                    
                    position = 0
                    if abs(pnl) > current_atr * 0.5:  # Only log significant trades
                        print(f"Exit Short #{trade_id}: {exit_price:.2f} ({exit_reason}) - P&L: {pnl:.2f} ({pnl_pct:.2f}%)")
            
            # Check for new entries (only when no position)
            if position == 0:
                if buy_signals[i]:
                    trade_id += 1
                    position = 1
                    entry_price = current_price
                    entry_time = current_time
                    # Set ATR-based hard stop
                    hard_stop_price = entry_price - (current_atr * self.hard_stop_atr_multiple)
                    print(f"Enter Long #{trade_id}: {entry_price:.2f} at {entry_time} (Stop: {hard_stop_price:.2f})")
                    
                elif sell_signals[i]:
                    trade_id += 1
                    position = -1
                    entry_price = current_price
                    entry_time = current_time
                    # Set ATR-based hard stop
                    hard_stop_price = entry_price + (current_atr * self.hard_stop_atr_multiple)
                    print(f"Enter Short #{trade_id}: {entry_price:.2f} at {entry_time} (Stop: {hard_stop_price:.2f})")
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(trades, initial_capital, equity, drawdown_curve)
        
        return {
            'trades': trades,
            'performance': performance,
            'equity_curve': drawdown_curve,
            'supertrend_data': st_df,
            'parameters': {
                'atr_period': self.atr_period,
                'factor': self.factor,
                'buffer_multiplier': self.buffer_multiplier,
                'hard_stop_atr_multiple': self.hard_stop_atr_multiple,
                'time_exit_hours': self.time_exit_hours
            }
        }
    
    def _apply_date_filter(self, df, start_date, end_date):
        """Apply date filtering to dataframe"""
        filtered_df = df.copy()
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df.index >= start_date]
            
        if end_date:
            end_date = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df.index <= end_date]
        
        if start_date or end_date:
            print(f"Date filter applied: {len(filtered_df)} rows (from {len(df)} original)")
            if len(filtered_df) > 0:
                print(f"Filtered date range: {filtered_df.index[0]} to {filtered_df.index[-1]}")
        
        return filtered_df
    
    def _calculate_performance_metrics(self, trades, initial_capital, final_equity, drawdown_curve):
        """Calculate comprehensive performance metrics"""
        if trades:
            profits = [t['pnl'] for t in trades]
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            
            total_return = (final_equity - initial_capital) / initial_capital * 100
            win_rate = len(winning_trades) / len(trades) * 100
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades else float('inf')
            max_drawdown = max(drawdown_curve) if drawdown_curve else 0
            avg_duration = np.mean([t['duration_hours'] for t in trades])
            
            # Calculate consecutive wins/losses
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            
            current_wins = 0
            current_losses = 0
            
            for trade in trades:
                if trade['pnl'] > 0:
                    current_wins += 1
                    current_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, current_wins)
                else:
                    current_losses += 1
                    current_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, current_losses)
            
            # Sharpe ratio (simplified)
            returns = [t['pnl_pct'] for t in trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
            
            # Exit reason analysis
            exit_reasons = {}
            for trade in trades:
                reason = trade['exit_reason']
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
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
                'max_consecutive_wins': max_consecutive_wins,
                'max_consecutive_losses': max_consecutive_losses,
                'sharpe_ratio': sharpe_ratio,
                'final_equity': final_equity,
                'exit_reasons': exit_reasons
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
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'sharpe_ratio': 0,
                'final_equity': initial_capital,
                'exit_reasons': {}
            }
        
        return performance

def load_data(file_path):
    """Load OHLC data from CSV file with proper date handling"""
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

def print_performance_summary(results):
    """Print detailed performance summary"""
    performance = results['performance']
    trades = results['trades']
    parameters = results['parameters']
    
    print("\n" + "=" * 80)
    print(" SUPERTREND ATR BUFFER STRATEGY - BACKTEST RESULTS ".center(80, "="))
    print("=" * 80)
    
    # Parameters
    print("\nStrategy Parameters:")
    print("-" * 40)
    print(f"ATR Period: {parameters['atr_period']}")
    print(f"SuperTrend Factor: {parameters['factor']}")
    print(f"Buffer Multiplier: {parameters['buffer_multiplier']}")
    print(f"Hard Stop (ATR Multiple): {parameters['hard_stop_atr_multiple']}")
    print(f"Time Exit (Hours): {parameters['time_exit_hours']} {'(disabled)' if parameters['time_exit_hours'] == 0 else ''}")
    
    # Performance metrics
    print("\nPerformance Summary:")
    print("-" * 40)
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
    print(f"Max Consecutive Wins: {performance['max_consecutive_wins']}")
    print(f"Max Consecutive Losses: {performance['max_consecutive_losses']}")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"Final Equity: {performance['final_equity']:.2f}")
    
    # Exit reason analysis
    if performance['exit_reasons']:
        print("\nExit Reason Analysis:")
        print("-" * 40)
        total_trades = sum(performance['exit_reasons'].values())
        for reason, count in performance['exit_reasons'].items():
            percentage = (count / total_trades) * 100
            print(f"{reason.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    # Recent trades sample
    if trades:
        print("\n" + "-" * 80)
        print(" RECENT TRADES SAMPLE ")
        print("-" * 80)
        
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
    
    print("=" * 80)

def save_results(results, filename):
    """Save backtest results to JSON file"""
    try:
        # Convert datetime objects to strings for JSON serialization
        results_copy = results.copy()
        
        # Convert trades
        for trade in results_copy['trades']:
            if isinstance(trade['entry_time'], pd.Timestamp):
                trade['entry_time'] = trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')
            if isinstance(trade['exit_time'], pd.Timestamp):
                trade['exit_time'] = trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')
        
        # Remove non-serializable data
        if 'supertrend_data' in results_copy:
            del results_copy['supertrend_data']
        
        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=4, default=str)
        
        print(f"\nResults saved to: {filename}")
        
    except Exception as e:
        print(f"Error saving results: {str(e)}")

def optimize_parameters(df, start_date=None, end_date=None):
    """
    Simple parameter optimization
    """
    print("\nRunning parameter optimization...")
    
    # Define parameter ranges to test
    atr_periods = [8, 10, 12, 14]
    factors = [2.5, 3.0, 3.5]
    buffer_multipliers = [0.2, 0.3, 0.4]
    hard_stop_multiples = [1.5, 2.0, 2.5]
    
    best_result = None
    best_sharpe = -999
    
    total_combinations = len(atr_periods) * len(factors) * len(buffer_multipliers) * len(hard_stop_multiples)
    print(f"Testing {total_combinations} parameter combinations...")
    
    current_combo = 0
    
    for atr_period in atr_periods:
        for factor in factors:
            for buffer_mult in buffer_multipliers:
                for hard_stop_mult in hard_stop_multiples:
                    current_combo += 1
                    
                    try:
                        # Create strategy with current parameters
                        strategy = SuperTrendStrategy(
                            atr_period=atr_period,
                            factor=factor,
                            buffer_multiplier=buffer_mult,
                            hard_stop_atr_multiple=hard_stop_mult,
                            time_exit_hours=0  # Disable time exit for optimization
                        )
                        
                        # Run backtest (suppress output)
                        original_stdout = sys.stdout
                        sys.stdout = open(os.devnull, 'w')
                        
                        result = strategy.backtest(df, start_date, end_date, initial_capital=10000)
                        
                        sys.stdout.close()
                        sys.stdout = original_stdout
                        
                        # Check if this is the best result
                        sharpe = result['performance']['sharpe_ratio']
                        if sharpe > best_sharpe and result['performance']['total_trades'] >= 10:
                            best_sharpe = sharpe
                            best_result = result
                        
                        # Progress update
                        if current_combo % 10 == 0:
                            print(f"Progress: {current_combo}/{total_combinations} ({current_combo/total_combinations*100:.1f}%)")
                        
                    except Exception as e:
                        print(f"Error testing combination {current_combo}: {str(e)}")
                        continue
    
    print(f"\nOptimization complete!")
    
    if best_result:
        print(f"Best parameters found:")
        params = best_result['parameters']
        print(f"  ATR Period: {params['atr_period']}")
        print(f"  Factor: {params['factor']}")
        print(f"  Buffer Multiplier: {params['buffer_multiplier']}")
        print(f"  Hard Stop Multiple: {params['hard_stop_atr_multiple']}")
        print(f"  Sharpe Ratio: {best_result['performance']['sharpe_ratio']:.3f}")
        print(f"  Total Return: {best_result['performance']['total_return']:.2f}%")
        print(f"  Win Rate: {best_result['performance']['win_rate']:.2f}%")
        return best_result
    else:
        print("No valid results found during optimization")
        return None

if __name__ == "__main__":
    import sys
    
    print("SuperTrend ATR Buffer Strategy - Improved Implementation")
    print("=" * 60)
    
    # Test with sample data file
    test_file = "/home/runner/work/STBacktest/STBacktest/btcusd_1m_20240401_to_20250606_gen20250607_233731.csv"
    
    if os.path.exists(test_file):
        print(f"Testing with: {os.path.basename(test_file)}")
        
        # Load data
        df = load_data(test_file)
        if df is not None:
            # Test with a sample (adjust size as needed)
            sample_size = 50000  # Increased sample size
            df_sample = df.tail(sample_size)
            print(f"Using sample of {len(df_sample)} bars for testing")
            
            # Create strategy with improved parameters
            strategy = SuperTrendStrategy(
                atr_period=14,           # Longer ATR period for stability
                factor=3.0,              # Standard factor
                buffer_multiplier=0.25,  # Smaller buffer for more selective entries
                hard_stop_atr_multiple=2.0,  # 2x ATR for stop loss
                time_exit_hours=48       # 48-hour maximum hold time
            )
            
            # Run backtest with date filtering (test recent period)
            print("\nRunning backtest with improved parameters...")
            results = strategy.backtest(
                df_sample,
                start_date='2025-05-01',  # Test recent period
                end_date=None,
                initial_capital=10000
            )
            
            # Display results
            print_performance_summary(results)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_results(results, f"supertrend_backtest_{timestamp}.json")
            
            # Ask if user wants to run optimization
            run_optimization = input("\nRun parameter optimization? (y/n): ").lower().strip()
            if run_optimization == 'y':
                # Use smaller sample for optimization to speed up
                opt_sample = df_sample.tail(20000)
                best_result = optimize_parameters(opt_sample, start_date='2025-05-01')
                
                if best_result:
                    print("\nOptimal parameters performance:")
                    print_performance_summary(best_result)
                    save_results(best_result, f"supertrend_optimized_{timestamp}.json")
            
        else:
            print("Failed to load test data")
    else:
        print(f"Test file not found: {test_file}")
        print("Please update the test_file path to point to your OHLC data")