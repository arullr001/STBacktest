##########Part 1 - Initial Setup and Utility Functions############
# Standard library imports
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import glob
import time
import traceback
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import logging
from pathlib import Path
import math
import psutil
import gc
from itertools import product

# Performance optimization imports
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client, LocalCluster
import jax.numpy as jnp
from jax import jit, vmap
import ray
import modin.pandas as mpd
import vaex
import cudf
from rapids import cupy as cp
from numba import cuda, float64, int32, jit
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Constants
THREADS_PER_BLOCK = 512  # Increased from 256
MAX_RAM_USAGE_PERCENT = 90
BATCH_SIZE = 1000
CURRENT_UTC = "2025-05-05 05:21:56"
CURRENT_USER = "arullr001"

# Fixed Parameter Ranges
PERIOD_RANGE = list(range(10, 101, 5))  # 10 to 100, step 5
MULTIPLIER_RANGE = [round(x, 2) for x in np.arange(3.0, 20.01, 0.01)]  # 3 to 20, step 0.01
PRICE_RANGE = [round(x, 2) for x in np.arange(0.05, 0.34, 0.01)]  # 0.05 to 0.33, step 0.01
TARGET_RANGE_MIN = 0.33
TARGET_RANGE_MAX = 1.1

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


############Part 2 - Parameter Handling and Data Loading##################


def get_parameter_inputs():
    """Get all parameter inputs from user with validation"""
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

        # Generate the ranges
        periods = list(range(period_start, period_end + 1, period_step))
        multipliers = [round(x, 2) for x in np.arange(mult_start, mult_end + (mult_step / 2), mult_step)]
        price_ranges = [round(x, 2) for x in np.arange(price_start, price_end + (price_step / 2), price_step)]

        # Calculate total combinations
        total_combinations = len(periods) * len(multipliers) * len(price_ranges)

        # Summary
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
        estimated_memory_mb = total_combinations * 0.5
        print(f"Estimated memory required: ~{estimated_memory_mb:.1f} MB")
        print(f"Available memory: {mem.available / (1024**2):.1f} MB")

        if total_combinations > 100000:
            print("\nWARNING: Very high number of combinations may cause long processing time")

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
    """Load and prepare OHLC data from file"""
    print(f"Attempting to load data from {file_path}...")

    try:
        if file_path.endswith('.csv'):
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
                print(f"Combining '{date_col}' and '{time_col}' into datetime")
                try:
                    df['datetime'] = pd.to_datetime(df[date_col] + ' ' + df[time_col])
                except Exception as e:
                    print(f"Error combining date and time: {e}")
                    try:
                        df['datetime'] = pd.to_datetime(df[date_col]) + pd.to_timedelta(df[time_col])
                    except Exception as e2:
                        print(f"Second attempt failed: {e2}")
                        print("Sample date values:", df[date_col].head())
                        print("Sample time values:", df[time_col].head())
            elif date_col:
                print(f"Using '{date_col}' as datetime")
                df['datetime'] = pd.to_datetime(df[date_col])

            # Check for required columns
            required_cols = ['open', 'high', 'low', 'close', 'datetime']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"Missing required columns: {missing_cols}")
                print("Available columns:", list(df.columns))
                raise ValueError(f"Could not find all required columns. Missing: {missing_cols}")

        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Set datetime as index
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        df.dropna(subset=numeric_cols, inplace=True)

        print(f"Successfully loaded data with shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")

        return df

    except Exception as e:
        print(f"Error during data loading: {str(e)}")
        raise

class DirectoryManager:
    """Manages directory structure for the application"""
    def __init__(self):
        current_file = os.path.basename(__file__)
        file_name = os.path.splitext(current_file)[0]
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_dir = f"{file_name}_{self.timestamp}"
        self.csv_dumps_dir = os.path.join(self.base_dir, "csv_dumps")
        self.error_logs_dir = os.path.join(self.base_dir, "error_logs")
        self.final_results_dir = os.path.join(self.base_dir, "final_results")
        self.create_directory_structure()
        self.setup_logging()

    def create_directory_structure(self):
        """Creates the necessary directory structure"""
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            os.makedirs(self.csv_dumps_dir, exist_ok=True)
            os.makedirs(self.error_logs_dir, exist_ok=True)
            os.makedirs(self.final_results_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory structure: {str(e)}")
            raise

    def setup_logging(self):
        """Sets up logging configuration"""
        try:
            # Configure processing errors logging
            processing_logger = logging.getLogger('processing_errors')
            processing_logger.setLevel(logging.ERROR)
            processing_handler = logging.FileHandler(
                os.path.join(self.error_logs_dir, 'processing_errors.log')
            )
            processing_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            processing_logger.addHandler(processing_handler)

            # Configure system errors logging
            system_logger = logging.getLogger('system_errors')
            system_logger.setLevel(logging.ERROR)
            system_handler = logging.FileHandler(
                os.path.join(self.error_logs_dir, 'system_errors.log')
            )
            system_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            system_logger.addHandler(system_handler)

        except Exception as e:
            print(f"Error setting up logging: {str(e)}")
            raise
            
            
            
####################Part 3 - GPU Acceleration and Calculation Components##################

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
    """GPU-accelerated SuperTrend calculation using CUDA"""
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
        gc.collect()
    except Exception as e:
        print(f"Warning: Could not clear GPU memory: {e}", end='\r')

    return df

@numba.jit(nopython=True)
def calculate_supertrend_numba(high, low, close, period, multiplier, price_range_1):
    """Optimized Numba implementation of SuperTrend calculation for CPU"""
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

def calculate_supertrend_cpu(df, period, multiplier, price_range_1):
    """CPU version of SuperTrend calculation"""
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

    # Generate signals
    df['buy_signal'] = (df['low'] < df['trailing_up_30']) & (df['close'] > df['trailing_up_30']) & (df['trend'] == 1)
    df['sell_signal'] = df['trend'] == -1
    df['short_signal'] = (df['high'] > df['trailing_dn_30']) & (df['close'] < df['trailing_dn_30']) & (df['trend'] == -1)
    df['cover_signal'] = df['trend'] == 1

    return df

def calculate_supertrend(df, period, multiplier, price_range_1):
    """Wrapper function that chooses between CPU or GPU implementation"""
    try:
        if cuda.is_available():
            try:
                device = cuda.get_current_device()
                try:
                    free_mem = device.mem_info()[0]
                except AttributeError:
                    try:
                        free_mem = device.memory_info().free
                    except AttributeError:
                        try:
                            ctx = cuda.current_context()
                            free_mem = ctx.get_memory_info().free
                        except AttributeError:
                            print("Unable to determine GPU memory, falling back to CPU", end='\r')
                            return calculate_supertrend_cpu(df, period, multiplier, price_range_1)

                # Calculate required memory
                data_size = len(df) * 8 * 5  # Approximate size in bytes
                
                if free_mem > data_size * 3:
                    print("Using GPU acceleration for SuperTrend calculation", end='\r')
                    return calculate_supertrend_gpu(df, period, multiplier, price_range_1)
                else:
                    print("Not enough GPU memory, falling back to CPU", end='\r')
                    return calculate_supertrend_cpu(df, period, multiplier, price_range_1)
            
            except Exception as e:
                print(f"GPU initialization failed, falling back to CPU: {str(e)}", end='\r')
                return calculate_supertrend_cpu(df, period, multiplier, price_range_1)
        
        return calculate_supertrend_cpu(df, period, multiplier, price_range_1)
    
    except Exception as e:
        print(f"Error in SuperTrend calculation, falling back to CPU: {str(e)}", end='\r')
        return calculate_supertrend_cpu(df, period, multiplier, price_range_1)




################Part 4 - Backtesting and Performance Analysis Components#############



def backtest_supertrend(df, period, multiplier, price_range_1, long_target=0.66, short_target=0.66):
    """Optimized backtesting of the Supertrend strategy"""
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
                if sell_signals[i]:
                    exit_condition = True
            else:
                long_target_price = entry_price * long_target_factor
                if current_price >= long_target_price:
                    exit_condition = True
                    exit_price = long_target_price
                elif sell_signals[i]:
                    exit_condition = True

            if exit_condition:
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
                if cover_signals[i]:
                    exit_condition = True
            else:
                short_target_price = entry_price * short_target_factor
                if current_price <= short_target_price:
                    exit_condition = True
                    exit_price = short_target_price
                elif cover_signals[i]:
                    exit_condition = True

            if exit_condition:
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

        # Check for new entries
        if not in_long and not in_short:
            if buy_signals[i]:
                trade_number += 1
                in_long = True
                entry_price = current_price
                entry_time = current_time
            elif short_signals[i]:
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

def cleanup_gpu_memory():
    """Clean up GPU memory to prevent leaks"""
    try:
        if cuda.is_available():
            gc.collect()
            cuda.current_context().deallocations.clear()
            print("GPU memory cleaned", end='\r')
    except Exception as e:
        print(f"GPU memory cleanup error: {e}", end='\r')

class BatchProcessor:
    """Handles batched processing of parameter combinations"""
    def __init__(self, directory_manager, memory_manager):
        self.dir_manager = directory_manager
        self.mem_manager = memory_manager
        self.current_batch = 0
        self.processing_logger = logging.getLogger('processing_errors')
        self.system_logger = logging.getLogger('system_errors')
        self.current_utc = CURRENT_UTC
        self.user = CURRENT_USER

    def save_batch_results(self, results, batch_num):
        """Saves batch results to CSV"""
        batch_file = os.path.join(self.dir_manager.csv_dumps_dir, f'batch_{batch_num}.csv')
        metadata_file = os.path.join(self.dir_manager.csv_dumps_dir, f'batch_{batch_num}_metadata.json')
        
        try:
            # Convert results to DataFrame for easier CSV handling
            results_data = []
            for r in results:
                if r['trades']:  # Only include valid results
                    row = {
                        'period': r['parameters']['period'],
                        'multiplier': r['parameters']['multiplier'],
                        'price_range_1': r['parameters']['price_range_1'],
                        'long_target': r['parameters']['long_target'],
                        'short_target': r['parameters']['short_target'],
                        'total_profit': r['total_profit'],
                        'trade_count': r['trade_count'],
                        'win_rate': r['win_rate']
                    }
                    results_data.append(row)

            if results_data:
                df = pd.DataFrame(results_data)
                df.to_csv(batch_file, index=False)

                # Save metadata
                metadata = {
                    'batch_number': batch_num,
                    'processed_at': self.current_utc,
                    'processed_by': self.user,
                    'combinations_count': len(results_data),
                    'memory_usage_mb': self.mem_manager.get_current_ram_usage() / (1024 * 1024),
                    'batch_success': True
                }
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=4)
                
                return True
            return False

        except Exception as e:
            self.processing_logger.error(f"Error saving batch {batch_num}: {str(e)}")
            return False

    def process_batch(self, df, param_combinations, batch_start, batch_size, max_workers=4):
        """Processes a batch of parameter combinations"""
        batch_end = min(batch_start + batch_size, len(param_combinations))
        batch_combinations = param_combinations[batch_start:batch_end]
        
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                executor.submit(
                    process_param_combo,
                    (df.copy(), period, multiplier, price_range, lt, st)
                )
                for period, multiplier, price_range, lt, st in batch_combinations
            ]

            for future in tqdm(concurrent.futures.as_completed(tasks), 
                             total=len(tasks), 
                             desc=f"Processing batch {self.current_batch + 1}"):
                try:
                    result = future.result()
                    if result['total_profit'] != float('-inf'):
                        results.append(result)
                except Exception as e:
                    self.processing_logger.error(f"Error in batch {self.current_batch + 1}: {str(e)}")

        return results



############# Part 5 - Results Processing and Visualization Components ##################

CURRENT_UTC = "2025-05-05 05:30:58"
CURRENT_USER = "arullr001"

class ResultsManager:
    """Manages the generation and storage of results and analysis"""
    def __init__(self, directory_manager):
        self.dir_manager = directory_manager
        self.processing_logger = logging.getLogger('processing_errors')
        self.system_logger = logging.getLogger('system_errors')

    def create_performance_summary(self, final_results_df):
        """Creates and saves a performance summary of the results"""
        try:
            if final_results_df.empty:
                return

            summary = {
                'total_combinations_tested': len(final_results_df),
                'best_performance': {
                    'total_profit': float(final_results_df['total_profit'].max()),
                    'parameters': final_results_df.iloc[0].to_dict(),
                },
                'average_performance': {
                    'profit': float(final_results_df['total_profit'].mean()),
                    'trade_count': float(final_results_df['trade_count'].mean()),
                    'win_rate': float(final_results_df['win_rate'].mean()),
                },
                'profit_distribution': {
                    'min': float(final_results_df['total_profit'].min()),
                    'max': float(final_results_df['total_profit'].max()),
                    'median': float(final_results_df['total_profit'].median()),
                    'std': float(final_results_df['total_profit'].std()),
                },
                'generated_at': CURRENT_UTC,
                'generated_by': CURRENT_USER
            }

            # Save summary to JSON
            summary_file = os.path.join(self.dir_manager.final_results_dir, 'performance_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4)

            # Create visualization
            self.create_performance_visualizations(final_results_df)

        except Exception as e:
            self.processing_logger.error(f"Error creating performance summary: {str(e)}")

    def create_performance_visualizations(self, df):
        """Creates visualization plots for the results"""
        try:
            plt.style.use('seaborn')

            # Create figures directory
            figures_dir = os.path.join(self.dir_manager.final_results_dir, 'figures')
            os.makedirs(figures_dir, exist_ok=True)

            # 1. Profit Distribution
            plt.figure(figsize=(12, 6))
            plt.hist(df['total_profit'], bins=50, edgecolor='black')
            plt.title('Distribution of Total Profit')
            plt.xlabel('Total Profit')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(figures_dir, 'profit_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 2. Win Rate vs Profit Scatter Plot
            plt.figure(figsize=(12, 6))
            plt.scatter(df['win_rate'], df['total_profit'], alpha=0.5)
            plt.title('Win Rate vs Total Profit')
            plt.xlabel('Win Rate')
            plt.ylabel('Total Profit')
            plt.savefig(os.path.join(figures_dir, 'winrate_vs_profit.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 3. Parameter Impact on Profit
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Period vs Profit
            scatter1 = axes[0].scatter(df['period'], df['total_profit'], 
                                     c=df['win_rate'], cmap='viridis', alpha=0.6)
            axes[0].set_title('Period vs Profit')
            axes[0].set_xlabel('Period')
            axes[0].set_ylabel('Total Profit')
            plt.colorbar(scatter1, ax=axes[0], label='Win Rate')

            # Multiplier vs Profit
            scatter2 = axes[1].scatter(df['multiplier'], df['total_profit'], 
                                     c=df['win_rate'], cmap='viridis', alpha=0.6)
            axes[1].set_title('Multiplier vs Profit')
            axes[1].set_xlabel('Multiplier')
            axes[1].set_ylabel('Total Profit')
            plt.colorbar(scatter2, ax=axes[1], label='Win Rate')

            # Price Range vs Profit
            scatter3 = axes[2].scatter(df['price_range_1'], df['total_profit'], 
                                     c=df['win_rate'], cmap='viridis', alpha=0.6)
            axes[2].set_title('Price Range vs Profit')
            axes[2].set_xlabel('Price Range')
            axes[2].set_ylabel('Total Profit')
            plt.colorbar(scatter3, ax=axes[2], label='Win Rate')

            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'parameter_impact.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 4. Correlation Heatmap
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[['period', 'multiplier', 'price_range_1', 
                                   'total_profit', 'trade_count', 'win_rate']].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Parameter Correlation Matrix')
            plt.savefig(os.path.join(figures_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 5. Trade Count Distribution
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='period', y='trade_count', data=df)
            plt.title('Trade Count Distribution by Period')
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(figures_dir, 'trade_count_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.processing_logger.error(f"Error creating visualizations: {str(e)}")

    def save_top_combinations_trades(self, final_results_df, original_data, top_n=10):
        """Saves detailed trade information for top performing parameter combinations"""
        try:
            if final_results_df.empty:
                return

            top_combinations = final_results_df.head(top_n)
            detailed_results_dir = os.path.join(self.dir_manager.final_results_dir, 'detailed_results')
            os.makedirs(detailed_results_dir, exist_ok=True)

            for idx, row in top_combinations.iterrows():
                try:
                    # Run backtest for this combination
                    result = backtest_supertrend(
                        original_data,
                        period=int(row['period']),
                        multiplier=row['multiplier'],
                        price_range_1=row['price_range_1'],
                        long_target=row['long_target'],
                        short_target=row['short_target']
                    )

                    # Convert trades to DataFrame for easier analysis
                    trades_df = pd.DataFrame(result['trades'])
                    
                    if not trades_df.empty:
                        # Calculate additional metrics
                        trades_df['holding_time'] = pd.to_datetime(trades_df['exit_time']) - pd.to_datetime(trades_df['entry_time'])
                        trades_df['profitable'] = trades_df['points'] > 0

                        # Save trades
                        filename = f'trades_p{int(row["period"])}_m{row["multiplier"]:.2f}_r{row["price_range_1"]:.2f}.csv'
                        trades_df.to_csv(os.path.join(detailed_results_dir, filename), index=False)

                        # Calculate and save trade statistics
                        stats = {
                            'parameters': row.to_dict(),
                            'trade_statistics': {
                                'total_trades': len(trades_df),
                                'profitable_trades': int(trades_df['profitable'].sum()),
                                'average_profit': float(trades_df['points'].mean()),
                                'max_profit': float(trades_df['points'].max()),
                                'max_loss': float(trades_df['points'].min()),
                                'profit_factor': float(
                                    trades_df[trades_df['points'] > 0]['points'].sum() /
                                    abs(trades_df[trades_df['points'] < 0]['points'].sum())
                                    if len(trades_df[trades_df['points'] < 0]) > 0 else float('inf')
                                ),
                                'average_holding_time': str(trades_df['holding_time'].mean()),
                                'long_trades': len(trades_df[trades_df['direction'] == 'long']),
                                'short_trades': len(trades_df[trades_df['direction'] == 'short'])
                            }
                        }

                        # Save statistics
                        stats_filename = f'stats_p{int(row["period"])}_m{row["multiplier"]:.2f}_r{row["price_range_1"]:.2f}.json'
                        with open(os.path.join(detailed_results_dir, stats_filename), 'w') as f:
                            json.dump(stats, f, indent=4)

                except Exception as e:
                    self.processing_logger.error(
                        f"Error processing detailed results for combination {row.to_dict()}: {str(e)}"
                    )

        except Exception as e:
            self.processing_logger.error(f"Error saving top combinations trades: {str(e)}")


##########Part 6 - Main Execution Flow and Program Entry Point##############

CURRENT_UTC = "2025-05-05 05:32:31"
CURRENT_USER = "arullr001"

def process_param_combo(args):
    """Process a single parameter combination"""
    try:
        df, period, multiplier, price_range, lt, st = args
        result = backtest_supertrend(
            df, 
            period=period,
            multiplier=multiplier,
            price_range_1=price_range,
            long_target=lt,
            short_target=st
        )
        return result
    except Exception as e:
        logging.getLogger('processing_errors').error(
            f"Error processing combination (period={period}, mult={multiplier}, "
            f"range={price_range}, lt={lt}, st={st}): {str(e)}"
        )
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

class SupertrendOptimizer:
    """Main class coordinating the optimization process"""
    def __init__(self):
        self.dir_manager = DirectoryManager()
        self.mem_manager = MemoryManager()
        self.data_loader = OptimizedDataLoader()
        self.distributed_computing = DistributedComputing()
        self.gpu_calculator = GPUAcceleratedCalculations(self.mem_manager)
        self.param_handler = ParameterHandler()
        self.results_processor = EnhancedResultsProcessor(self.dir_manager, self.mem_manager)
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Setup logging configuration"""
        logger = logging.getLogger('supertrend_optimizer')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        file_handler = logging.FileHandler(
            Path(self.dir_manager.error_logs_dir) / 'optimization.log'
        )
        console_handler = logging.StreamHandler()
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def run_optimization(self, data_file):
        """Main optimization process"""
        try:
            self.logger.info("Starting optimization process")
            
            # 1. Initialize distributed computing
            self.logger.info("Initializing distributed computing")
            self.distributed_computing.initialize_dask()
            self.distributed_computing.initialize_ray()
            
            # 2. Load and prepare data
            self.logger.info(f"Loading data from {data_file}")
            df = self.data_loader.load_data(data_file)
            
            # 3. Get target inputs
            self.logger.info("Getting target inputs")
            long_target, short_target = self.param_handler.get_target_inputs()
            
            # 4. Generate parameter combinations
            self.logger.info("Generating parameter combinations")
            combinations = self.param_handler.generate_parameter_combinations(
                long_target, short_target
            )
            
            # 5. Estimate processing time
            estimate = self.param_handler.estimate_processing_time(
                len(combinations),
                self.distributed_computing.n_workers
            )
            self.logger.info(
                f"Estimated processing time: {estimate['estimated_time']['minutes']:.2f} minutes"
            )
            
            # 6. Process combinations in batches
            results = self._process_combinations(df, combinations)
            
            # 7. Process and visualize results
            self.logger.info("Processing results")
            analysis = self.results_processor.process_results(results)
            
            # 8. Create visualizations
            self.logger.info("Creating visualizations")
            self.results_processor.create_visualizations(results)
            
            # 9. Save detailed results
            self.logger.info("Saving detailed results")
            self.results_processor.save_detailed_results(
                results,
                self._get_trades_for_top_combinations(df, results)
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Optimization process failed: {str(e)}")
            raise
        finally:
            self.cleanup()

    def _process_combinations(self, df, combinations):
        """Process parameter combinations using distributed computing"""
        total_combinations = len(combinations)
        batch_size = self.mem_manager.optimize_batch_size(
            df.memory_usage().sum(),
            total_combinations
        )
        
        results_list = []
        
        with tqdm(total=total_combinations, desc="Processing combinations") as pbar:
            for i in range(0, total_combinations, batch_size):
                batch = combinations[i:i + batch_size]
                
                # Process batch using Ray
                @ray.remote
                def process_batch(batch_combinations):
                    return [
                        self._process_single_combination(df, *params)
                        for params in batch_combinations
                    ]
                
                batch_results = ray.get(process_batch.remote(batch))
                results_list.extend(batch_results)
                
                # Update progress
                pbar.update(len(batch))
                
                # Cleanup after batch
                self.mem_manager.cleanup()
        
        return pd.DataFrame(results_list)

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.distributed_computing.cleanup()
            self.mem_manager.cleanup()
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")

def main():
    """Main execution function"""
    try:
        start_time = time.time()
        print("=" * 50)
        print(" SUPER TREND STRATEGY BACKTESTER (OPTIMIZED) ".center(50, "="))
        print("=" * 50)
        print(f"Started at (UTC): {CURRENT_UTC}")
        print(f"User: {CURRENT_USER}")

        # Initialize optimizer
        optimizer = SupertrendOptimizer()
        
        # Get data file
        print("\nEnter the path to your OHLC data file:")
        data_file = input("> ").strip()
        
        if not os.path.exists(data_file):
            print("Error: File not found")
            return
            
        # Run optimization
        analysis = optimizer.run_optimization(data_file)
        
        # Display results
        print("\nOptimization completed!")
        print("\nTop 3 Parameter Combinations:")
        for idx, combo in enumerate(analysis['top_3_combinations'], 1):
            print(f"\n{idx}. Combination:")
            print(f"   Period: {combo['period']}")
            print(f"   Multiplier: {combo['multiplier']:.2f}")
            print(f"   Price Range: {combo['price_range_1']:.2f}")
            print(f"   Total Profit: {combo['total_profit']:.2f}")
            print(f"   Win Rate: {combo['win_rate']*100:.2f}%")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.getLogger('system_errors').error(f"Fatal error: {str(e)}\n{traceback.format_exc()}")
    finally:
        cleanup_gpu_memory()
