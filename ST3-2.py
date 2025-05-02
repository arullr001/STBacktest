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

# Define constants for GPU threads
THREADS_PER_BLOCK = 256

# Function to find OHLC files
def find_ohlc_files(data_dir):
    """
    Find all OHLC data files in the given directory
    """
    files = []
    supported_extensions = ['.csv', '.xlsx', '.xls']
    for ext in supported_extensions:
        files.extend(glob.glob(os.path.join(data_dir, f'*{ext}')))
    return files

# Function to process user-selected files
def select_files_to_process(data_files):
    """
    Let user select which files to process
    """
    if not data_files:
        print("No data files found")
        return []

    print(f"Found {len(data_files)} data files for testing")
    print("\nAvailable OHLC files:")
    for i, file_path in enumerate(data_files):
        print(f"[{i+1}] {os.path.basename(file_path)}")

    print("\nEnter file number(s) to process (comma-separated, or 'all' for all files):")
    selection = input("> ").strip().lower()

    selected_files = []
    if selection == 'all':
        selected_files = data_files
    else:
        try:
            # Parse user selection
            indices = [int(idx.strip()) - 1 for idx in selection.split(',') if idx.strip()]
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
    
    
# Utility function to get user input for a float value
def get_float_input(prompt, min_value=None, max_value=None, allow_zero=False):
    """
    Get validated float input from user
    """
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

# Utility function to get user input for an integer value
def get_int_input(prompt, min_value=None, max_value=None):
    """
    Get validated integer input from user
    """
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

# Function to get parameter inputs from the user
def get_parameter_inputs():
    """
    Get all parameter inputs from user with validation
    """
    while True:
        print("\n" + "=" * 50)
        print(" PARAMETER RANGE CONFIGURATION ".center(50, "="))
        print("=" * 50)

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
        multipliers = [round(x, 2) for x in np.arange(mult_start, mult_end + (mult_step / 2), mult_step)]
        price_ranges = [round(x, 2) for x in np.arange(price_start, price_end + (price_step / 2), price_step)]

        # Calculate total combinations
        total_combinations = len(periods) * len(multipliers) * len(price_ranges)

        # Summarize the inputs and ask for confirmation
        print("\n" + "=" * 50)
        print(" PARAMETER SUMMARY ".center(50, "="))
        print("=" * 50)
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
        
        
# Function to load OHLC data from a file
def load_ohlc_data(file_path):
    """
    Load and prepare OHLC data from file - improved version with better column detection and datetime handling
    """
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
    """
    CUDA kernel for SuperTrend calculation
    """
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
    blocks_per_grid = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    # Launch CUDA kernel
    calculate_supertrend_cuda_kernel[blocks_per_grid, THREADS_PER_BLOCK](
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
    """
    CPU version of SuperTrend calculation (original implementation)
    """
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
    Optimized Numba implementation of SuperTrend calculation for CPU
    """
    n = len(close)
    up = np.zeros(n, dtype=np.float64)
    dn = np.zeros(n, dtype=np.float64)
    trend = np.zeros(n, dtype=np.int64)
    trailing_up_30 = np.zeros(n, dtype=np.float64)
    trailing_dn_30 = np.zeros(n, dtype=np.float64)

    for i in range(n):
        hl2 = (high[i] + low[i]) / 2

        # Calculate TR and ATR
        if i == 0:
            tr = high[0] - low[0]
            atr = tr
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr = (atr * (period - 1) + tr) / period

        # Determine basic up and down levels
        up_basic = hl2 - (multiplier * atr)
        dn_basic = hl2 + (multiplier * atr)

        if i == 0:
            up[i] = up_basic
            dn[i] = dn_basic
            trend[i] = 0
        else:
            # Update up and down levels
            up[i] = max(up_basic, up[i-1]) if close[i-1] > up[i-1] else up_basic
            dn[i] = min(dn_basic, dn[i-1]) if close[i-1] < dn[i-1] else dn_basic

            # Update trend
            if close[i] > dn[i-1]:
                trend[i] = 1  # Uptrend
            elif close[i] < up[i-1]:
                trend[i] = -1  # Downtrend
            else:
                trend[i] = trend[i-1]  # Maintain previous trend

        # Calculate trailing levels
        trailing_up_30[i] = up[i] + up[i] * (price_range_1 / 100)
        trailing_dn_30[i] = dn[i] - dn[i] * (price_range_1 / 100)

    return up, dn, trend, trailing_up_30, trailing_dn_30
    
    
def cleanup_gpu_memory():
    """
    Clean up GPU memory to prevent leaks
    """
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

    long_target_factor = 1 + long_target / 100 if not use_trend_exit_for_long else None
    short_target_factor = 1 - short_target / 100 if not use_trend_exit_for_short else None

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
    
    
def optimize_parameters(df, params, max_workers=4):
    """
    Optimize parameters using multiprocessing.

    Args:
        df: DataFrame with OHLC data
        params: Dictionary of ranges for optimization
        max_workers: Maximum number of concurrent workers

    Returns:
        List of results sorted by total profit
    """
    # Extract parameter ranges
    periods = params['periods']
    multipliers = params['multipliers']
    price_ranges = params['price_ranges']
    long_target = params['long_target']
    short_target = params['short_target']

    # Generate all combinations of parameters
    param_combinations = list(product(periods, multipliers, price_ranges))

    print(f"\nOptimizing over {len(param_combinations)} parameter combinations...")

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use partial to pass other arguments
        tasks = [
            executor.submit(
                process_param_combo,
                (df, period, multiplier, price_range, long_target, short_target)
            )
            for period, multiplier, price_range in param_combinations
        ]

        for future in tqdm(concurrent.futures.as_completed(tasks), total=len(tasks), desc="Processing"):
            try:
                result = future.result()
                if result['total_profit'] != float('-inf'):
                    results.append(result)
            except Exception as e:
                print(f"Error processing combination: {e}")

    # Sort results by total profit
    sorted_results = sorted(results, key=lambda x: x['total_profit'], reverse=True)

    print(f"\nOptimization complete. Top results:")
    for rank, result in enumerate(sorted_results[:5]):
        params = result['parameters']
        print(f"[{rank + 1}] Profit: {result['total_profit']:.2f}, Trades: {result['trade_count']}, "
              f"Win Rate: {result['win_rate']:.2%} - Params: {params}")

    return sorted_results
    
    
def main():
    """
    Main function to run the Supertrend strategy backtester and optimizer
    """
    print("=" * 50)
    print(" SUPER TREND STRATEGY BACKTESTER ".center(50, "="))
    print("=" * 50)

    # Step 1: Find OHLC files
    data_dir = input("\nEnter the directory containing OHLC data files: ").strip()
    data_files = find_ohlc_files(data_dir)

    if not data_files:
        print("No data files found in the specified directory. Exiting.")
        return

    # Step 2: Let user select files to process
    selected_files = select_files_to_process(data_files)
    if not selected_files:
        print("No files selected for processing. Exiting.")
        return

    # Step 3: Get parameter inputs from user
    params = get_parameter_inputs()

    # Step 4: Process each file
    for file_path in selected_files:
        print(f"\nProcessing file: {file_path}")
        try:
            # Load OHLC data
            df = load_ohlc_data(file_path)

            # Optimize parameters
            optimize_parameters(df, params)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            traceback.print_exc()

    print("\nAll files processed. Exiting.")


if __name__ == "__main__":
    main()