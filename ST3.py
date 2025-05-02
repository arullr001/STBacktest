import os
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
import json
import glob
import time
import traceback
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import numba  # Add numba for JIT compilation
import psutil  # For memory usage monitoring
import warnings
from concurrent.futures import ProcessPoolExecutor
import cupy as cp
import numba.cuda
from numba import cuda
import concurrent.futures


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def find_ohlc_files(data_dir):
    """Find all OHLC data files in the given directory"""
    files = []
    supported_extensions = ['.csv', '.xlsx', '.xls']
    
    for ext in supported_extensions:
        files.extend(glob.glob(os.path.join(data_dir, f'*{ext}')))
    
    return files

def select_files_to_process(data_files):
    """Let user select which files to process"""
    if not data_files:
        print("No data files found")
        return []
    
    print(f"Found {len(data_files)} data files for testing")
    
    # Display available files with numbering
    print("\nAvailable OHLC files:")
    for i, file_path in enumerate(data_files):
        print(f"[{i+1}] {os.path.basename(file_path)}")
    
    # Get user selection
    print("\nEnter file number(s) to process (comma-separated, or 'all' for all files):")
    selection = input("> ").strip().lower()
    
    selected_files = []
    if selection == 'all':
        selected_files = data_files
    else:
        try:
            # Parse user selection (1-based indices)
            indices = [int(idx.strip()) - 1 for idx in selection.split(',') if idx.strip()]
            
            # Validate indices
            valid_indices = [idx for idx in indices if 0 <= idx < len(data_files)]
            if not valid_indices:
                print("No valid file selections. Exiting.")
                return []
                
            selected_files = [data_files[idx] for idx in valid_indices]
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")
            return []
    
    print(f"\nProcessing {len(selected_files)} file(s)")
    return selected_files

def get_float_input(prompt, min_value=None, max_value=None, allow_zero=False):
    """Get validated float input from user"""
    while True:
        try:
            value = input(prompt).strip()
            if value == "" and allow_zero:
                return 0.0
            value = float(value)
            
            if min_value is not None and value < min_value:
                print(f"Value must be at least {min_value}")
                continue
                
            if max_value is not None and value > max_value:
                print(f"Value must be at most {max_value}")
                continue
                
            return value
        except ValueError:
            print("Please enter a valid number")

def get_int_input(prompt, min_value=None, max_value=None):
    """Get validated integer input from user"""
    while True:
        try:
            value = int(input(prompt).strip())
            
            if min_value is not None and value < min_value:
                print(f"Value must be at least {min_value}")
                continue
                
            if max_value is not None and value > max_value:
                print(f"Value must be at most {max_value}")
                continue
                
            return value
        except ValueError:
            print("Please enter a valid integer")

def get_parameter_inputs():
    """Get all parameter inputs from user with validation"""
    while True:
        print("\n" + "="*50)
        print(" PARAMETER RANGE CONFIGURATION ".center(50, "="))
        print("="*50)
        
        # Period range inputs
        print("\nEnter Period range (Integer values):")
        period_start = get_int_input("  Start value (min 5): ", 5)
        period_end = get_int_input(f"  End value (min {period_start}): ", period_start)
        period_step = get_int_input("  Step/increment (min 1): ", 1)
        
        # Multiplier range inputs
        print("\nEnter Multiplier range (Float values):")
        mult_start = get_float_input("  Start value (min 0.5): ", 0.5)
        mult_end = get_float_input(f"  End value (min {mult_start}): ", mult_start)
        mult_step = get_float_input("  Step/increment (min 0.01): ", 0.01)
        
        # Price range inputs
        print("\nEnter Price Range values (Float values):")
        price_start = get_float_input("  Start value (min 0.01): ", 0.01)
        price_end = get_float_input(f"  End value (min {price_start}): ", price_start)
        price_step = get_float_input("  Step/increment (min 0.01): ", 0.01)
        
        # Target inputs
        print("\nEnter Target values (enter 0 for trend-based exits):")
        long_target = get_float_input("  Long target percentage (0 for trend-based exit): ", 0, allow_zero=True)
        short_target = get_float_input("  Short target percentage (0 for trend-based exit): ", 0, allow_zero=True)
        
        # Generate the ranges to show the user how many combinations
        periods = list(range(period_start, period_end + 1, period_step))
        multipliers = [round(x, 2) for x in np.arange(mult_start, mult_end + (mult_step/2), mult_step)]
        price_ranges = [round(x, 2) for x in np.arange(price_start, price_end + (price_step/2), price_step)]
        
        # Calculate total combinations
        total_combinations = len(periods) * len(multipliers) * len(price_ranges)
        
        # Summarize the inputs and ask for confirmation
        print("\n" + "="*50)
        print(" PARAMETER SUMMARY ".center(50, "="))
        print("="*50)
        print(f"Period range: {period_start} to {period_end} (step {period_step}) - {len(periods)} values")
        print(f"Multiplier range: {mult_start} to {mult_end} (step {mult_step}) - {len(multipliers)} values")
        print(f"Price range: {price_start} to {price_end} (step {price_step}) - {len(price_ranges)} values")
        print(f"Long target: {long_target}% {'(trend-based exit)' if long_target == 0 else ''}")
        print(f"Short target: {short_target}% {'(trend-based exit)' if short_target == 0 else ''}")
        print(f"\nTotal parameter combinations to test: {total_combinations}")
        
        # Memory estimation
        mem = psutil.virtual_memory()
        estimated_memory_mb = total_combinations * 0.5  # Rough estimate: 0.5 MB per combination
        print(f"Estimated memory required: ~{estimated_memory_mb:.1f} MB")
        print(f"Available memory: {mem.available / (1024**2):.1f} MB")
        
        # Warning if combinations are very high
        if total_combinations > 100000:
            print("\nWARNING: Very high number of combinations may cause long processing time")
            
        # Ask for confirmation
        confirm = input("\nProceed with these parameters? (y/n): ").lower().strip()
        if confirm == 'y':
            return {
                'periods': periods,
                'multipliers': multipliers,
                'price_ranges': price_ranges,
                'long_target': long_target,
                'short_target': short_target,
                'total_combinations': total_combinations
            }
        print("\nLet's reconfigure the parameters...")

def load_ohlc_data(file_path):
    """Load and prepare OHLC data from file - improved version with better column detection and datetime handling"""
    print(f"Attempting to load data from {file_path}...")
    
    if file_path.endswith('.csv'):
        try:
            # Read a small sample first to examine the columns
            sample = pd.read_csv(file_path, nrows=5)
            print(f"File columns: {list(sample.columns)}")
            
            # Check for separate date and time columns
            has_date_col = any('date' in col.lower() for col in sample.columns)
            has_time_col = any('time' in col.lower() and 'datetime' not in col.lower() for col in sample.columns)
            
            # Read the full file
            df = pd.read_csv(file_path)
            
            # Map columns to standard names
            column_mapping = {}
            date_col = None
            time_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if col in ['O', 'o'] or 'open' in col_lower:
                    column_mapping[col] = 'open'
                elif col in ['H', 'h'] or 'high' in col_lower:
                    column_mapping[col] = 'high'
                elif col in ['L', 'l'] or 'low' in col_lower:
                    column_mapping[col] = 'low'
                elif col in ['C', 'c'] or 'close' in col_lower:
                    column_mapping[col] = 'close'
                elif 'date' in col_lower and 'datetime' not in col_lower:
                    date_col = col
                elif 'time' in col_lower and 'datetime' not in col_lower:
                    time_col = col
                elif any(x in col_lower for x in ['datetime']):
                    column_mapping[col] = 'datetime'
            
            print(f"Column mapping: {column_mapping}")
            print(f"Date column: {date_col}, Time column: {time_col}")
            
            # Apply column mapping for OHLC
            df = df.rename(columns=column_mapping)
            
            # Handle datetime creation
            if date_col and time_col:
                # Combine date and time columns into datetime
                print(f"Combining '{date_col}' and '{time_col}' into datetime")
                try:
                    # Try to convert to datetime directly
                    df['datetime'] = pd.to_datetime(df[date_col] + ' ' + df[time_col])
                except Exception as e:
                    print(f"Error combining date and time: {e}")
                    # Try alternative formats
                    try:
                        df['datetime'] = pd.to_datetime(df[date_col]) + pd.to_timedelta(df[time_col])
                    except Exception as e2:
                        print(f"Second attempt failed: {e2}")
                        # Show the date and time values for debugging
                        print("Sample date values:", df[date_col].head())
                        print("Sample time values:", df[time_col].head())
            elif date_col:
                # Only date column exists
                print(f"Using '{date_col}' as datetime")
                df['datetime'] = pd.to_datetime(df[date_col])
            
            # Check if we have all required columns
            required_cols = ['open', 'high', 'low', 'close', 'datetime']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"Missing required columns: {missing_cols}")
                print("Available columns:", list(df.columns))
                raise ValueError(f"Could not find all required columns. Missing: {missing_cols}")
            
        except Exception as e:
            print(f"Error during data loading: {str(e)}")
            raise
            
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
        # Similar column mapping would be applied here
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Convert numeric columns to proper types
    numeric_cols = ['open', 'high', 'low', 'close']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Set datetime as index
    df.set_index('datetime', inplace=True)
    
    # Sort by datetime
    df.sort_index(inplace=True)
    
    # Drop rows with NaN in critical columns
    df.dropna(subset=numeric_cols, inplace=True)
    
    print(f"Successfully loaded data with shape: {df.shape}")
    print(f"Index type: {type(df.index)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df
	



@cuda.jit
def calculate_supertrend_cuda_kernel(high, low, close, period, multiplier, price_range_1, up, dn, trend, trailing_up_30, trailing_dn_30):
    """CUDA kernel for SuperTrend calculation"""
    i = cuda.grid(1)
    if i >= len(close):
        return
        
    # Initialize thread-local variables
    hl2 = (high[i] + low[i]) / 2
    
    # Calculate TR for this point
    if i == 0:
        tr = high[0] - low[0]
    else:
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Calculate ATR for this point
    atr_sum = 0.0
    count = 0
    for j in range(max(0, i - period + 1), i + 1):
        if j == 0:
            atr_sum += high[0] - low[0]
        else:
            atr_sum += max(
                high[j] - low[j],
                abs(high[j] - close[j-1]),
                abs(low[j] - close[j-1])
            )
        count += 1
    atr = atr_sum / count
    
    # Calculate basic SuperTrend components
    up_basic = hl2 - (multiplier * atr)
    dn_basic = hl2 + (multiplier * atr)
    
    # Calculate SuperTrend values
    if i == 0:
        up[i] = up_basic
        dn[i] = dn_basic
        trend[i] = 0
    else:
        # Calculate up
        if close[i-1] > up[i-1]:
            up[i] = max(up_basic, up[i-1])
        else:
            up[i] = up_basic
        
        # Calculate down
        if close[i-1] < dn[i-1]:
            dn[i] = min(dn_basic, dn[i-1])
        else:
            dn[i] = dn_basic
        
        # Determine trend
        if close[i] > dn[i-1]:
            trend[i] = 1  # Uptrend
        elif close[i] < up[i-1]:
            trend[i] = -1  # Downtrend
        else:
            trend[i] = trend[i-1]  # Maintain previous trend
    
    # Calculate trailing levels
    trailing_up_30[i] = up[i] + up[i] * (price_range_1 / 100)
    trailing_dn_30[i] = dn[i] - dn[i] * (price_range_1 / 100)


def calculate_supertrend_gpu(df, period, multiplier, price_range_1):
    """
    GPU-accelerated SuperTrend calculation using CUDA.
    
    Args:
        df: DataFrame with OHLC data
        period: ATR period
        multiplier: ATR multiplier
        price_range_1: Super Trend Price Range - 1 value
        
    Returns:
        DataFrame with Supertrend indicators and signals
    """
    df = df.copy()
    
    # Extract arrays for GPU
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    n = len(close)
    
    # Convert to contiguous arrays for CUDA
    high_gpu = cuda.to_device(high.astype(np.float64))
    low_gpu = cuda.to_device(low.astype(np.float64))
    close_gpu = cuda.to_device(close.astype(np.float64))
    
    # Create output arrays on GPU
    up_gpu = cuda.device_array(n, dtype=np.float64)
    dn_gpu = cuda.device_array(n, dtype=np.float64)
    trend_gpu = cuda.device_array(n, dtype=np.float64)
    trailing_up_30_gpu = cuda.device_array(n, dtype=np.float64)
    trailing_dn_30_gpu = cuda.device_array(n, dtype=np.float64)
    
    # Configure CUDA kernel
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    
    # Launch CUDA kernel
    calculate_supertrend_cuda_kernel[blocks_per_grid, threads_per_block](
        high_gpu, low_gpu, close_gpu, period, multiplier, price_range_1,
        up_gpu, dn_gpu, trend_gpu, trailing_up_30_gpu, trailing_dn_30_gpu
    )
    
    # Copy results back from GPU
    up = up_gpu.copy_to_host()
    dn = dn_gpu.copy_to_host()
    trend = trend_gpu.copy_to_host()
    trailing_up_30 = trailing_up_30_gpu.copy_to_host()
    trailing_dn_30 = trailing_dn_30_gpu.copy_to_host()
    
    # Add results to DataFrame
    df['up'] = up
    df['dn'] = dn
    df['trend'] = trend
    df['trailing_up_30'] = trailing_up_30
    df['trailing_dn_30'] = trailing_dn_30
    
    # Generate signals based on the strategy rules - vectorized operations
    df['buy_signal'] = (df['low'] < df['trailing_up_30']) & (df['close'] > df['trailing_up_30']) & (df['trend'] == 1)
    df['sell_signal'] = df['trend'] == -1
    df['short_signal'] = (df['high'] > df['trailing_dn_30']) & (df['close'] < df['trailing_dn_30']) & (df['trend'] == -1)
    df['cover_signal'] = df['trend'] == 1
    
    # Clean up GPU memory
    try:
        # More robust memory cleanup
        del high_gpu, low_gpu, close_gpu, up_gpu, dn_gpu, trend_gpu, trailing_up_30_gpu, trailing_dn_30_gpu
        cuda.current_context().deallocations.clear()
        import gc
        gc.collect()
    except Exception as e:
        print(f"Warning: Could not clear GPU memory: {e}", end='\r')
    
    return df


def calculate_supertrend(df, period, multiplier, price_range_1):
    """
    Wrapper function that chooses between CPU or GPU implementation based on availability.
    
    Args:
        df: DataFrame with OHLC data
        period: ATR period
        multiplier: ATR multiplier
        price_range_1: Super Trend Price Range - 1 value
        
    Returns:
        DataFrame with Supertrend indicators and signals
    """
    # Check if CUDA is available
    try:
        if cuda.is_available():
            # Get device memory info
            device = cuda.get_current_device()
            free_mem = device.memory_info().free if hasattr(device, 'memory_info') else device.total_memory
            data_size = len(df) * 8 * 5  # Approximate size in bytes (8 bytes per double × 5 arrays)
        
            # Only use GPU if enough memory is available
            if free_mem > data_size * 3:  # Allow buffer
                print("Using GPU acceleration for SuperTrend calculation", end='\r')
                return calculate_supertrend_gpu(df, period, multiplier, price_range_1)
            else:
                print("Not enough GPU memory, falling back to CPU")
        # If we get here, use CPU version
        return calculate_supertrend_cpu(df, period, multiplier, price_range_1)
    except Exception as e:
        print(f"GPU acceleration failed, falling back to CPU: {str(e)}")
        return calculate_supertrend_cpu(df, period, multiplier, price_range_1)


def calculate_supertrend_cpu(df, period, multiplier, price_range_1):
    """CPU version of SuperTrend calculation (original implementation)"""
    df = df.copy()
    
    # Extract numpy arrays for Numba function
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Calculate using Numba function
    up, dn, trend, trailing_up_30, trailing_dn_30 = calculate_supertrend_numba(
        high, low, close, period, multiplier, price_range_1
    )
    
    # Add results to DataFrame
    df['up'] = up
    df['dn'] = dn
    df['trend'] = trend
    df['trailing_up_30'] = trailing_up_30
    df['trailing_dn_30'] = trailing_dn_30
    
    # Generate signals based on the strategy rules - vectorized operations
    df['buy_signal'] = (df['low'] < df['trailing_up_30']) & (df['close'] > df['trailing_up_30']) & (df['trend'] == 1)
    df['sell_signal'] = df['trend'] == -1
    df['short_signal'] = (df['high'] > df['trailing_dn_30']) & (df['close'] < df['trailing_dn_30']) & (df['trend'] == -1)
    df['cover_signal'] = df['trend'] == 1
    
    return df


@numba.jit(nopython=True)
def calculate_supertrend_numba(high, low, close, period, multiplier, price_range_1):
    """
    Calculate Supertrend indicator using Numba for acceleration.
    
    Args:
        high: numpy array of high prices
        low: numpy array of low prices
        close: numpy array of close prices
        period: ATR period
        multiplier: ATR multiplier
        price_range_1: Super Trend Price Range - 1 value
    
    Returns:
        up, dn, trend, trailing_up_30, trailing_dn_30 arrays
    """
    n = len(close)
    
    # Calculate HL2
    hl2 = (high + low) / 2
    
    # Calculate TR and ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]  # Initial value
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Calculate ATR with simple moving average
    atr = np.zeros(n)
    for i in range(n):
        if i < period:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = np.mean(tr[i-period+1:i+1])
    
    # Basic Supertrend components
    up_basic = hl2 - (multiplier * atr)
    dn_basic = hl2 + (multiplier * atr)
    
    # Initialize arrays
    up = np.zeros(n)
    dn = np.zeros(n)
    trend = np.zeros(n)
    
    # First values
    up[0] = up_basic[0]
    dn[0] = dn_basic[0]
    trend[0] = 0
    
    # Calculate Supertrend
    for i in range(1, n):
        # Calculate up
        if close[i-1] > up[i-1]:
            up[i] = max(up_basic[i], up[i-1])
        else:
            up[i] = up_basic[i]
        
        # Calculate down
        if close[i-1] < dn[i-1]:
            dn[i] = min(dn_basic[i], dn[i-1])
        else:
            dn[i] = dn_basic[i]
        
        # Determine trend
        if close[i] > dn[i-1]:
            trend[i] = 1  # Uptrend
        elif close[i] < up[i-1]:
            trend[i] = -1  # Downtrend
        else:
            trend[i] = trend[i-1]  # Maintain previous trend
    
    # Calculate trailing levels
    trailing_up_30 = up + up * (price_range_1 / 100)
    trailing_dn_30 = dn - dn * (price_range_1 / 100)
    
    return up, dn, trend, trailing_up_30, trailing_dn_30



def backtest_supertrend(df, period, multiplier, price_range_1, long_target=0.66, short_target=0.66):
    """
    Optimized backtesting of the Supertrend strategy
    
    Args:
        df: DataFrame with OHLC data
        period: ATR period
        multiplier: ATR multiplier
        price_range_1: Super Trend Price Range - 1 value
        long_target: Profit target percentage for long positions
        short_target: Profit target percentage for short positions
        
    Returns:
        Dictionary with backtest results and trade list
    """
    # Calculate Supertrend and signals
    st_df = calculate_supertrend(df, period, multiplier, price_range_1)
    
    # Pre-compute target factors for efficiency
    use_trend_exit_for_long = long_target <= 0
    use_trend_exit_for_short = short_target <= 0
    
    long_target_factor = 1 + long_target/100 if not use_trend_exit_for_long else None
    short_target_factor = 1 - short_target/100 if not use_trend_exit_for_short else None
    
    # Convert to numpy arrays for faster operations
    close = st_df['close'].values
    buy_signals = st_df['buy_signal'].values
    sell_signals = st_df['sell_signal'].values
    short_signals = st_df['short_signal'].values
    cover_signals = st_df['cover_signal'].values
    dates = st_df.index.to_numpy()
    
    # Initialize trade tracking variables
    trades = []
    trade_number = 0
    in_long = False
    in_short = False
    entry_price = 0
    entry_time = None
    
    # Loop through data for backtesting - optimized version
    for i in range(1, len(st_df)):
        current_price = close[i]
        current_time = dates[i]
        
        # Handle existing long position
        if in_long:
            exit_condition = False
            exit_price = current_price
            
            if use_trend_exit_for_long:
                # Exit on trend change
                if sell_signals[i]:
                    exit_condition = True
            else:
                # Exit on target or trend change
                long_target_price = entry_price * long_target_factor
                if current_price >= long_target_price:
                    exit_condition = True
                    exit_price = long_target_price
                elif sell_signals[i]:
                    exit_condition = True
            
            if exit_condition:
                # Record the trade
                trades.append({
                    'trade_number': trade_number,
                    'direction': 'long',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': current_time,
                    'exit_price': exit_price,
                    'points': exit_price - entry_price
                })
                
                in_long = False
        
        # Handle existing short position
        if in_short:
            exit_condition = False
            exit_price = current_price
            
            if use_trend_exit_for_short:
                # Exit on trend change
                if cover_signals[i]:
                    exit_condition = True
            else:
                # Exit on target or trend change
                short_target_price = entry_price * short_target_factor
                if current_price <= short_target_price:
                    exit_condition = True
                    exit_price = short_target_price
                elif cover_signals[i]:
                    exit_condition = True
            
            if exit_condition:
                # Record the trade
                trades.append({
                    'trade_number': trade_number,
                    'direction': 'short',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': current_time,
                    'exit_price': exit_price,
                    'points': entry_price - exit_price
                })
                
                in_short = False
        
        # Check for new long entry if not already in position
        if not in_long and not in_short and buy_signals[i]:
            trade_number += 1
            in_long = True
            entry_price = current_price
            entry_time = current_time
        
        # Check for new short entry if not already in position
        elif not in_long and not in_short and short_signals[i]:
            trade_number += 1
            in_short = True
            entry_price = current_price
            entry_time = current_time
    
        # Close any open position at the end of testing
    if in_long:
        trades.append({
            'trade_number': trade_number,
            'direction': 'long',
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': dates[-1],
            'exit_price': close[-1],
            'points': close[-1] - entry_price
        })
    
    if in_short:
        trades.append({
            'trade_number': trade_number,
            'direction': 'short',
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': dates[-1],
            'exit_price': close[-1],
            'points': entry_price - close[-1]
        })
     
    # Calculate performance metrics
    total_profit = sum(trade['points'] for trade in trades) if trades else 0
    win_trades = [trade for trade in trades if trade['points'] > 0]
    win_rate = len(win_trades) / len(trades) if trades else 0
    
    return {
        'parameters': {
            'period': period,
            'multiplier': multiplier,
            'price_range_1': price_range_1,
            'long_target': long_target,
            'short_target': short_target
        },
        'total_profit': total_profit,
        'trade_count': len(trades),
        'win_rate': win_rate,
        'trades': trades
    }


    

def backtest_supertrend_cpu(df, period, multiplier, price_range_1, long_target=0.66, short_target=0.66):
    """
    CPU-only version of backtesting the Supertrend strategy to avoid CUDA issues in multiprocessing
    
    Args:
        df: DataFrame with OHLC data
        period: ATR period
        multiplier: ATR multiplier
        price_range_1: Super Trend Price Range - 1 value
        long_target: Profit target percentage for long positions
        short_target: Profit target percentage for short positions
        
    Returns:
        Dictionary with backtest results and trade list
    """
    # Calculate Supertrend using CPU-only implementation
    st_df = calculate_supertrend_cpu(df, period, multiplier, price_range_1)
    
    # Pre-compute target factors for efficiency
    use_trend_exit_for_long = long_target <= 0
    use_trend_exit_for_short = short_target <= 0
    
    long_target_factor = 1 + long_target/100 if not use_trend_exit_for_long else None
    short_target_factor = 1 - short_target/100 if not use_trend_exit_for_short else None
    
    # Convert to numpy arrays for faster operations
    close = st_df['close'].values
    buy_signals = st_df['buy_signal'].values
    sell_signals = st_df['sell_signal'].values
    short_signals = st_df['short_signal'].values
    cover_signals = st_df['cover_signal'].values
    dates = st_df.index.to_numpy()
    
    # Initialize trade tracking variables
    trades = []
    trade_number = 0
    in_long = False
    in_short = False
    entry_price = 0
    entry_time = None
    
    # Loop through data for backtesting - optimized version
    for i in range(1, len(st_df)):
        current_price = close[i]
        current_time = dates[i]
        
        # Handle existing long position
        if in_long:
            exit_condition = False
            exit_price = current_price
            
            if use_trend_exit_for_long:
                # Exit on trend change
                if sell_signals[i]:
                    exit_condition = True
            else:
                # Exit on target or trend change
                long_target_price = entry_price * long_target_factor
                if current_price >= long_target_price:
                    exit_condition = True
                    exit_price = long_target_price
                elif sell_signals[i]:
                    exit_condition = True
            
            if exit_condition:
                # Record the trade
                trades.append({
                    'trade_number': trade_number,
                    'direction': 'long',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': current_time,
                    'exit_price': exit_price,
                    'points': exit_price - entry_price
                })
                
                in_long = False
        
        # Handle existing short position
        if in_short:
            exit_condition = False
            exit_price = current_price
            
            if use_trend_exit_for_short:
                # Exit on trend change
                if cover_signals[i]:
                    exit_condition = True
            else:
                # Exit on target or trend change
                short_target_price = entry_price * short_target_factor
                if current_price <= short_target_price:
                    exit_condition = True
                    exit_price = short_target_price
                elif cover_signals[i]:
                    exit_condition = True
            
            if exit_condition:
                # Record the trade
                trades.append({
                    'trade_number': trade_number,
                    'direction': 'short',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': current_time,
                    'exit_price': exit_price,
                    'points': entry_price - exit_price
                })
                
                in_short = False
        
        # Check for new long entry if not already in position
        if not in_long and not in_short and buy_signals[i]:
            trade_number += 1
            in_long = True
            entry_price = current_price
            entry_time = current_time
        
        # Check for new short entry if not already in position
        elif not in_long and not in_short and short_signals[i]:
            trade_number += 1
            in_short = True
            entry_price = current_price
            entry_time = current_time
    
    # Close any open position at the end of testing
    if in_long:
        trades.append({
            'trade_number': trade_number,
            'direction': 'long',
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': dates[-1],
            'exit_price': close[-1],
            'points': close[-1] - entry_price
        })
    
    if in_short:
        trades.append({
            'trade_number': trade_number,
            'direction': 'short',
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': dates[-1],
            'exit_price': close[-1],
            'points': entry_price - close[-1]
        })
     
    # Calculate performance metrics
    total_profit = sum(trade['points'] for trade in trades) if trades else 0
    win_trades = [trade for trade in trades if trade['points'] > 0]
    win_rate = len(win_trades) / len(trades) if trades else 0
    
    return {
        'parameters': {
            'period': period,
            'multiplier': multiplier,
            'price_range_1': price_range_1,
            'long_target': long_target,
            'short_target': short_target
        },
        'total_profit': total_profit,
        'trade_count': len(trades),
        'win_rate': win_rate,
        'trades': trades
    }




def test_param_combination(args):
    """
    Test a single parameter combination with unpacked arguments for parallel processing.
    
    Args:
        args: Tuple containing (df, period, multiplier, price_range_1, long_target, short_target)
        
    Returns:
        Backtest results dictionary
    """
    df, period, multiplier, price_range_1, long_target, short_target = args
    try:
        # Use CPU version directly to avoid CUDA context issues in multiprocessing
        return backtest_supertrend_cpu(
            df, 
            period=period,
            multiplier=multiplier,
            price_range_1=price_range_1,
            long_target=long_target,
            short_target=short_target
        )
    except Exception as e:
        # Return a fallback object if the combination causes errors
        print(f"Error in parameter combination: period={period}, multiplier={multiplier:.2f}, "
              f"price_range_1={price_range_1:.2f} - {str(e)}")
        return {
            'parameters': {
                'period': period,
                'multiplier': multiplier,
                'price_range_1': price_range_1,
                'long_target': long_target,
                'short_target': short_target
            },
            'total_profit': float('-inf'),
            'trade_count': 0,
            'win_rate': 0,
            'trades': []
        }



    
def cleanup_gpu_memory():
    """Clean up GPU memory to prevent leaks"""
    try:
        if cuda.is_available():
            # Explicitly force garbage collection
            import gc
            gc.collect()
            # Clear CUDA cache
            cuda.current_context().deallocations.clear()
            print("GPU memory cleaned", end='\r')
    except Exception as e:
        print(f"GPU memory cleanup error: {e}", end='\r')        





def process_param_combo(args):
    """
    Process a single parameter combination for backtesting.
    
    Args:
        args: Tuple containing (df, period, multiplier, price_range, long_target, short_target)
        
    Returns:
        Dictionary with backtest results
    """
    df, period, multiplier, price_range, lt, st = args
    try:
        return backtest_supertrend_cpu(
            df, period=period, multiplier=multiplier, 
            price_range_1=price_range, long_target=lt, short_target=st
        )
    except Exception as e:
        print(f"Error in parameter combination: period={period}, multiplier={multiplier:.2f}, "
              f"price_range_1={price_range:.2f} - {str(e)}")
        return {
            'parameters': {
                'period': period,
                'multiplier': multiplier,
                'price_range_1': price_range,
                'long_target': lt,
                'short_target': st
            },
            'total_profit': float('-inf'),
            'trade_count': 0,
            'win_rate': 0,
            'trades': []
        }




def optimize_parameters(df, param_inputs, output_filename):
    """
    Optimized parameter search using user-defined parameter ranges.
    
    Args:
        df: DataFrame with OHLC data
        param_inputs: Dictionary containing parameter ranges
        output_filename: Base name for output files
        
    Returns:
        List of top 3 parameter sets with results
    """
    print(f"Starting parameter optimization for {output_filename}...")
    
    # Extract parameter ranges from inputs
    periods = param_inputs['periods']
    multipliers = param_inputs['multipliers']
    price_ranges = param_inputs['price_ranges']
    long_target = param_inputs['long_target']
    short_target = param_inputs['short_target']
    
    total_combinations = param_inputs['total_combinations']
    
    print(f"- Periods: {len(periods)} values ({periods[0]} to {periods[-1]})")
    print(f"- Multipliers: {len(multipliers)} values ({multipliers[0]} to {multipliers[-1]:.2f})")
    print(f"- Price Ranges: {len(price_ranges)} values ({price_ranges[0]} to {price_ranges[-1]:.2f})")
    print(f"- Long Target: {long_target}% {'(trend-based exit)' if long_target <= 0 else ''}")
    print(f"- Short Target: {short_target}% {'(trend-based exit)' if short_target <= 0 else ''}")
    
    print(f"Total parameter combinations to test: {total_combinations}")
    
    # Optimize CPU and memory usage
    available_memory = psutil.virtual_memory().available
    data_size = df.memory_usage(deep=True).sum()
    required_memory_per_process = data_size * 2  # Reduced from 3 to 2
    
    # Calculate optimal number of processes - more conservative
    max_processes_by_memory = max(1, int(available_memory * 0.5 / required_memory_per_process))
    max_processes_by_cores = max(1, mp.cpu_count() - 2)  # Leave two cores free
    
    num_processes = min(max_processes_by_memory, max_processes_by_cores, 6)  # Cap at 6 processes
    print(f"Using {num_processes} processes for parallel computation")
    
    # Generate parameter combinations 
    param_combinations = list(product(periods, multipliers, price_ranges))
    
    # Calculate optimal chunk size
    chunksize = max(1, min(500, total_combinations // (num_processes * 20)))
    
    # Run parameter search
    all_results = []
    start_time = time.time()
    
    # To avoid memory issues with multiprocessing, we'll use a simpler approach
    # Process combinations in sequential batches
    batch_size = 1000
    
    for i in range(0, len(param_combinations), batch_size):
        batch = param_combinations[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(param_combinations) + batch_size - 1)//batch_size}")
        
        # Create args for this batch
        batch_args = [(df, period, multiplier, price_range, long_target, short_target) 
                     for period, multiplier, price_range in batch]
        
        # Use the ProcessPoolExecutor for this batch
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            batch_results = list(tqdm(
                executor.map(process_param_combo, batch_args, chunksize=chunksize),
                total=len(batch),
                desc=f"Batch {i//batch_size + 1}"
            ))
        
        # Add to results
        all_results.extend(batch_results)
        
        # Force garbage collection
        import gc
        gc.collect()
    
    total_time = time.time() - start_time
    print(f"Parameter testing completed in {total_time:.2f} seconds")
    
    # Sort results by total profit
    all_results.sort(key=lambda x: x['total_profit'], reverse=True)
    
    print(f"Tested {len(all_results)} parameter combinations")
    
    # Force garbage collection
    import gc
    gc.collect()
    
    return all_results[:3]







def save_results(top_results, output_dir, filename_base):
    """
    Optimized function to save and visualize optimization results.
    
    Args:
        top_results: List of top parameter results
        output_dir: Directory to save results
        filename_base: Base name for output files
        
    Returns:
        Path to the result directory
    """
    # Create output directory with timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, f"{filename_base}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"Saving results to {result_dir}...")
    
    # Create a summary DataFrame for all top results
    summary_data = []
    for i, result in enumerate(top_results):
        params = result['parameters']
        summary_data.append({
            'rank': i+1,
            'period': params['period'],
            'multiplier': round(params['multiplier'], 2),
            'price_range_1': round(params['price_range_1'], 2),
            'long_target': params['long_target'],
            'short_target': params['short_target'],
            'total_profit': round(result['total_profit'], 2),
            'trade_count': result['trade_count'],
            'win_rate': round(result['win_rate'] * 100, 1),
            'avg_profit_per_trade': round(result['total_profit'] / result['trade_count'], 2) if result['trade_count'] > 0 else 0,
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary as CSV
    summary_csv_path = os.path.join(result_dir, 'parameter_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    
    # Create enhanced visualizations of the results
    try:
        # 1. Bar chart of total profits with error bars for better visualization
        plt.figure(figsize=(10, 6))
        plt.bar(
            [f"Rank {i+1}" for i in range(len(top_results))],
            [r['total_profit'] for r in top_results],
            yerr=[r['total_profit'] * (1-r['win_rate']) / 2 for r in top_results],
            capsize=5,
            color='skyblue',
            edgecolor='navy'
        )
        plt.title(f"Top Parameter Sets - Total Profit Comparison", fontsize=14)
        plt.ylabel("Total Profit", fontsize=12)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
		
        # Add value labels on top of bars
        for i, result in enumerate(top_results):
            plt.text(i, result['total_profit'] + 0.1, 
                    f"{result['total_profit']:.1f}", 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'profit_comparison.png'), dpi=300)
        plt.close()
        
        # 2. Multi-metric comparison chart
        metrics = ['total_profit', 'win_rate', 'trade_count']
        labels = [f"Rank {i+1}" for i in range(len(top_results))]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Profit
        axes[0].bar(labels, [r['total_profit'] for r in top_results], color='green')
        axes[0].set_title('Total Profit')
        axes[0].grid(alpha=0.3)
        
        # Win Rate
        axes[1].bar(labels, [r['win_rate'] * 100 for r in top_results], color='blue')
        axes[1].set_title('Win Rate (%)')
        axes[1].grid(alpha=0.3)
        
        # Trade Count
        axes[2].bar(labels, [r['trade_count'] for r in top_results], color='orange')
        axes[2].set_title('Number of Trades')
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'multi_metric_comparison.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create visualization - {str(e)}")
        
        
        # Create parameter-specific directories and detailed results
    for i, result in enumerate(top_results):
        rank = i + 1
        param_dir = os.path.join(result_dir, f"rank_{rank}_params")
        os.makedirs(param_dir, exist_ok=True)
        
        # Save parameter details as JSON
        params = result['parameters']
        with open(os.path.join(param_dir, 'parameters.json'), 'w') as f:
            json.dump({
                'period': params['period'],
                'multiplier': params['multiplier'],
                'price_range_1': params['price_range_1'],
                'long_target': params['long_target'],
                'short_target': params['short_target'],
                'total_profit': result['total_profit'],
                'trade_count': result['trade_count'],
                'win_rate': result['win_rate'],
            }, f, indent=4)
			
        # Save trade list if there are trades
        if result['trades']:
            # Create a DataFrame from trades list
            trades_df = pd.DataFrame(result['trades'])
            
            # Convert datetime objects to strings for CSV
            if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                trades_df['entry_time'] = trades_df['entry_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
                trades_df['exit_time'] = trades_df['exit_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            
            # Add calculated fields
            trades_df['profit_pct'] = trades_df.apply(
                lambda row: (row['points'] / row['entry_price']) * 100 if row['entry_price'] > 0 else 0, 
                axis=1
            )
            
            # Mark winning and losing trades
            trades_df['is_win'] = trades_df['points'] > 0
            
            # Save trade details
            trades_df.to_csv(os.path.join(param_dir, 'trades.csv'), index=False)
            
            # Create enhanced trade analysis visualizations
            try:
                # Cumulative profit chart
                plt.figure(figsize=(12, 6))
                cumulative_profit = trades_df['points'].cumsum()
                plt.plot(range(len(trades_df)), cumulative_profit, 
                        marker='o', markersize=4, linestyle='-', linewidth=2, color='blue')
                
                # Add horizontal line at 0
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                
                plt.title(f"Cumulative Profit - Rank {rank} Parameters", fontsize=14)
                plt.xlabel("Trade Number", fontsize=12)
                plt.ylabel("Cumulative Points", fontsize=12)
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(param_dir, 'cumulative_profit.png'), dpi=300)
                plt.close()
                
            except Exception as e:
                print(f"Warning: Could not create trade visualizations for rank {rank} - {str(e)}")
    
    print(f"Results saved successfully to {result_dir}")
    return result_dir
    

def main():
    """
    Main execution function for Supertrend parameter optimization.
    With user-defined parameter ranges and improved resource management.
    """
    print("\n" + "="*70)
    print(" SUPERTREND STRATEGY PARAMETER OPTIMIZATION (OPTIMIZED VERSION) ".center(70, "="))
    print("="*70 + "\n")
   
    # Check GPU availability
    try:
        if cuda.is_available():
            device = cuda.get_current_device()
            print(f"CUDA GPU detected: {device.name}")
            print(f"Compute capability: {device.compute_capability}")
            
            # Different ways to get memory info depending on CUDA/CuPy version
            try:
                total_mem = device.total_memory
                print(f"Total memory: {total_mem / (1024**3):.2f} GB")
            except AttributeError:
                try:
                    mem_info = cuda.current_context().get_memory_info()
                    print(f"Free memory: {mem_info[0] / (1024**3):.2f} GB / Total: {mem_info[1] / (1024**3):.2f} GB")
                except Exception:
                    print("Could not determine GPU memory info")
        else:
            print("No CUDA-capable GPU detected. Using CPU only.")
    except Exception as e:
        print(f"Error checking GPU: {str(e)}")
        print("Continuing with CPU only.")
   
    # Define directories
    data_dir = "data"  # Directory containing OHLC data files
    output_dir = "results"  # Directory for saving results
    
    print(f"Looking for data files in: {os.path.abspath(data_dir)}")
    print(f"Results will be saved to: {os.path.abspath(output_dir)}")
	
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all data files
    print("Scanning for OHLC data files...")
    data_files = find_ohlc_files(data_dir)
    
    if not data_files:
        print(f"No data files found in {data_dir}. Please ensure your data is in the correct directory.")
        return
    
    # Let user select which files to process
    selected_files = select_files_to_process(data_files)
    
    if not selected_files:
        print("No files selected for processing. Exiting.")
        return
    
    # Get parameter inputs from user
    param_inputs = get_parameter_inputs()
    
    # Process each selected data file
    overall_start_time = time.time()
    results_summary = []
    
    for file_idx, file_path in enumerate(selected_files):
        try:
            filename = os.path.basename(file_path)
            filename_base = os.path.splitext(filename)[0]
            
            print("\n" + "="*70)
            print(f" Processing file {file_idx+1}/{len(selected_files)}: {filename} ".center(70, "="))
            print("="*70)
            
            file_start_time = time.time()
            
            # Display system resources
            mem = psutil.virtual_memory()
            print(f"Available memory: {mem.available / (1024**3):.1f} GB / {mem.total / (1024**3):.1f} GB")
            
            # Load data
            print(f"Loading data from {filename}...")
            try:
                df = load_ohlc_data(file_path)
                print(f"Successfully loaded {len(df)} data points from {filename}")
                
                # Display dataset information
                try:
                    date_range = f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
                except AttributeError:
                    date_range = f"{df.index.min()} to {df.index.max()}"
                
                print(f"Date range: {date_range}")
                print(f"Data frequency: {pd.infer_freq(df.index) or 'Unknown/Irregular'}")
                print(f"Price range: {df['low'].min():.2f} to {df['high'].max():.2f}")
                
            except Exception as e:
                print(f"Error loading data: {str(e)}")
                traceback.print_exc()
                continue
            
            # Optimize parameters
            try:
                top_results = optimize_parameters(df, param_inputs, filename_base)
                if not top_results:
                    print(f"No valid parameter sets found for {filename}. Skipping to next file.")
                    continue
            except Exception as e:
                print(f"Error during parameter optimization: {str(e)}")
                traceback.print_exc()
                continue
            
            # Save results
            try:
                result_dir = save_results(top_results, output_dir, filename_base)
            except Exception as e:
                print(f"Error saving results: {str(e)}")
                traceback.print_exc()
                result_dir = "Results not saved due to error"
                
            # Print best parameters
            print("\nTop parameter sets:")
            for i, result in enumerate(top_results):
                params = result['parameters']
                print(f"Rank {i+1}: Period={params['period']}, Multiplier={params['multiplier']:.2f}, "
                      f"Price Range={params['price_range_1']:.2f} → Profit: {result['total_profit']:.2f}, "
                      f"Trades: {result['trade_count']}, Win Rate: {result['win_rate']*100:.1f}%")
            
            # Add to summary
            elapsed = time.time() - file_start_time
            results_summary.append({
                'filename': filename,
                'data_points': len(df),
                'date_range': date_range,
                'best_period': top_results[0]['parameters']['period'],
                'best_multiplier': top_results[0]['parameters']['multiplier'],
                'best_price_range': top_results[0]['parameters']['price_range_1'],
                'best_profit': top_results[0]['total_profit'],
                'best_trades': top_results[0]['trade_count'],
                'best_win_rate': top_results[0]['win_rate'],
                'processing_time': elapsed,
                'result_dir': result_dir
            })
            
            print(f"\nCompleted processing {filename} in {elapsed:.1f} seconds")
            print(f"Results saved to: {result_dir}")
            
        except Exception as e:
            print(f"Unexpected error processing {file_path}:")
            traceback.print_exc()
    
    # Create an overall summary if multiple files were processed
    if len(results_summary) > 1:
        try:
            summary_df = pd.DataFrame(results_summary)
            summary_path = os.path.join(output_dir, f"overall_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"\nOverall summary saved to: {summary_path}")
        except Exception as e:
            print(f"Error creating overall summary: {str(e)}")
    
    # Print overall time
    total_elapsed = time.time() - overall_start_time
    print(f"\nTotal execution time for all files: {total_elapsed:.1f} seconds")
    print("\nSupertrend optimization completed!")

if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        traceback.print_exc()
    finally:
        elapsed = time.time() - start_time
        print(f"\nTotal execution time: {elapsed:.1f} seconds")