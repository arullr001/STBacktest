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
import numba
import psutil
import warnings
from concurrent.futures import ProcessPoolExecutor
import cupy as cp
import numba.cuda
from numba import cuda
import concurrent.futures
import logging
from pathlib import Path
import math
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define constants
THREADS_PER_BLOCK = 256
MAX_RAM_USAGE_PERCENT = 90
BATCH_SIZE = 1000
CURRENT_UTC = "2025-05-03 03:57:20"  # Updated timestamp
CURRENT_USER = "arullr001"           # Updated user

# Utility Functions
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

def get_parameter_inputs():
    """Get all parameter inputs from user with validation"""
    while True:
        print("\n" + "=" * 50)
        print(" PARAMETER RANGE CONFIGURATION ".center(50, "="))
        print("=" * 50)

        # ATR Length range inputs
        print("\nEnter ATR Length range (Integer values):")
        atr_length_start = get_int_input("  Start value (min 1): ", 1)
        atr_length_end = get_int_input(f"  End value (min {atr_length_start}): ", atr_length_start)
        atr_length_step = get_int_input("  Step/increment (min 1): ", 1)

        # Supertrend Factor range inputs
        print("\nEnter Supertrend Factor range (Float values):")
        factor_start = get_float_input("  Start value (min 0.5): ", 0.5)
        factor_end = get_float_input(f"  End value (min {factor_start}): ", factor_start)
        factor_step = get_float_input("  Step/increment (min 0.1): ", 0.1)

        # ATR Buffer Multiplier range inputs
        print("\nEnter ATR Buffer Multiplier range (Float values):")
        buffer_start = get_float_input("  Start value (min 0.1): ", 0.1)
        buffer_end = get_float_input(f"  End value (min {buffer_start}): ", buffer_start)
        buffer_step = get_float_input("  Step/increment (min 0.1): ", 0.1)

        # Hard Stop Distance inputs
        print("\nEnter Hard Stop Distance range (Integer values):")
        stop_start = get_int_input("  Start value (min 1): ", 1)
        stop_end = get_int_input(f"  End value (min {stop_start}): ", stop_start)
        stop_step = get_int_input("  Step/increment (min 1): ", 1)

        # Generate the ranges
        atr_lengths = list(range(atr_length_start, atr_length_end + 1, atr_length_step))
        factors = [round(x, 2) for x in np.arange(factor_start, factor_end + (factor_step / 2), factor_step)]
        buffers = [round(x, 2) for x in np.arange(buffer_start, buffer_end + (buffer_step / 2), buffer_step)]
        stops = list(range(stop_start, stop_end + 1, stop_step))

        # Calculate total combinations
        total_combinations = len(atr_lengths) * len(factors) * len(buffers) * len(stops)

        # Summary
        print("\n" + "=" * 50)
        print(" PARAMETER SUMMARY ".center(50, "="))
        print("=" * 50)
        print(f"ATR Length range: {atr_length_start} to {atr_length_end} (step {atr_length_step}) - {len(atr_lengths)} values")
        print(f"Factor range: {factor_start} to {factor_end} (step {factor_step}) - {len(factors)} values")
        print(f"Buffer range: {buffer_start} to {buffer_end} (step {buffer_step}) - {len(buffers)} values")
        print(f"Hard Stop range: {stop_start} to {stop_end} (step {stop_step}) - {len(stops)} values")
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
                'atr_lengths': atr_lengths,
                'factors': factors,
                'buffers': buffers,
                'stops': stops,
                'total_combinations': total_combinations
            }
        print("\nLet's reconfigure the parameters...")

        
# First, update the timestamp constants
CURRENT_UTC = "2025-05-03 20:22:21"
CURRENT_USER = "arullr001"

class DirectoryManager:
    """Manages directory structure for the application"""
    def __init__(self):
        # Get the name of the currently executing Python file
        current_file = os.path.basename(__file__)  # Gets 'ST.py'
        file_name = os.path.splitext(current_file)[0]  # Removes '.py' extension
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Simplified directory name: just filename_timestamp
        self.base_dir = f"{file_name}_{self.timestamp}"
        self.csv_dumps_dir = os.path.join(self.base_dir, "csv_dumps")
        self.error_logs_dir = os.path.join(self.base_dir, "error_logs")
        self.final_results_dir = os.path.join(self.base_dir, "final_results")
        self.create_directory_structure()
        self.setup_logging()

    def create_directory_structure(self):
        """Creates the necessary directory structure for the application"""
        try:
            # Create main directories
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
            # Configure logging for processing errors
            processing_logger = logging.getLogger('processing_errors')
            processing_logger.setLevel(logging.ERROR)
            processing_handler = logging.FileHandler(
                os.path.join(self.error_logs_dir, 'processing_errors.log')
            )
            processing_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            processing_logger.addHandler(processing_handler)

            # Configure logging for system errors
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

class MemoryManager:
    """Manages memory usage and monitoring"""
    def __init__(self, max_ram_percent=MAX_RAM_USAGE_PERCENT):
        self.max_ram_percent = max_ram_percent
        self.total_ram = psutil.virtual_memory().total
        self.max_ram_bytes = (self.total_ram * self.max_ram_percent) / 100

    def get_current_ram_usage(self):
        """Returns current RAM usage in bytes"""
        return psutil.Process().memory_info().rss

    def get_available_ram(self):
        """Returns available RAM in bytes"""
        return self.max_ram_bytes - self.get_current_ram_usage()

    def is_ram_available(self, required_bytes):
        """Checks if there's enough RAM available"""
        return self.get_available_ram() >= required_bytes

    def estimate_combination_memory(self, df_size, n_combinations):
        """Estimates memory required for processing combinations"""
        return df_size * 2 * n_combinations

    def calculate_optimal_batch_size(self, df_size, total_combinations):
        """Calculates optimal batch size based on available memory"""
        available_ram = self.get_available_ram()
        estimated_mem_per_combo = df_size * 2
        optimal_batch_size = int(available_ram / (estimated_mem_per_combo * 1.2))
        return min(optimal_batch_size, BATCH_SIZE, total_combinations)

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
        
        
# Update the timestamp constant
CURRENT_UTC = "2025-05-03 04:01:10"
CURRENT_USER = "arullr001"


def calculate_supertrend_cpu(df, atr_length, factor, buffer_multiplier):
    """CPU version of SuperTrend calculation"""
    df = df.copy()
    
    # Calculate ATR
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    tr = np.zeros(len(df))
    atr = np.zeros(len(df))
    
    # Calculate TR and ATR
    for i in range(len(df)):
        if i == 0:
            tr[i] = high[i] - low[i]
            atr[i] = tr[i]
        else:
            tr[i] = max(high[i] - low[i],
                       abs(high[i] - close[i-1]),
                       abs(low[i] - close[i-1]))
            atr[i] = (atr[i-1] * (atr_length - 1) + tr[i]) / atr_length
    
    # Calculate basic bands
    hl2 = (high + low) / 2
    up = hl2 - (factor * atr)
    dn = hl2 + (factor * atr)
    
    # Initialize trend arrays
    trend = np.zeros(len(df))
    trend_up = np.zeros(len(df))
    trend_down = np.zeros(len(df))
    
    # Calculate SuperTrend
    for i in range(1, len(df)):
        if close[i-1] > trend_up[i-1]:
            trend_up[i] = max(up[i], trend_up[i-1])
        else:
            trend_up[i] = up[i]
            
        if close[i-1] < trend_down[i-1]:
            trend_down[i] = min(dn[i], trend_down[i-1])
        else:
            trend_down[i] = dn[i]
            
        if close[i] > trend_down[i-1]:
            trend[i] = 1  # Uptrend
        elif close[i] < trend_up[i-1]:
            trend[i] = -1  # Downtrend
        else:
            trend[i] = trend[i-1]  # Maintain previous trend
    
    # Add results to DataFrame
    df['up'] = trend_up
    df['dn'] = trend_down
    df['trend'] = trend
    
    # Calculate buffer zones
    df['trailing_up'] = df['up'] + (df['up'] * buffer_multiplier)
    df['trailing_dn'] = df['dn'] - (df['dn'] * buffer_multiplier)
    
    # Generate signals based on the strategy rules
    df['buy_signal'] = (df['trend'] == 1) & (df['low'] < df['trailing_up']) & (df['close'] > df['trailing_up'])
    df['sell_signal'] = df['trend'] == -1
    df['short_signal'] = (df['trend'] == -1) & (df['high'] > df['trailing_dn']) & (df['close'] < df['trailing_dn'])
    df['cover_signal'] = df['trend'] == 1
    
    return df

@cuda.jit
def calculate_supertrend_cuda_kernel(high, low, close, atr_length, factor, buffer_multiplier,
                                   up, dn, trend, trailing_up, trailing_dn):
    """CUDA kernel for SuperTrend calculation"""
    i = cuda.grid(1)
    if i >= len(close):
        return

    # Initialize thread-local variables
    hl2 = (high[i] + low[i]) / 2

    # Calculate TR and ATR
    if i == 0:
        tr = high[0] - low[0]
        atr = tr
    else:
        tr = max(high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1]))
        atr = (atr * (atr_length - 1) + tr) / atr_length

    # Calculate basic bands
    up_basic = hl2 - (factor * atr)
    dn_basic = hl2 + (factor * atr)

    if i == 0:
        up[i] = up_basic
        dn[i] = dn_basic
        trend[i] = 0
    else:
        # Calculate up trend
        if close[i-1] > up[i-1]:
            up[i] = max(up_basic, up[i-1])
        else:
            up[i] = up_basic

        # Calculate down trend
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
            trend[i] = trend[i-1]

    # Calculate buffer zones
    trailing_up[i] = up[i] + (up[i] * buffer_multiplier)
    trailing_dn[i] = dn[i] - (dn[i] * buffer_multiplier)

def calculate_supertrend_gpu(df, atr_length, factor, buffer_multiplier):
    """GPU-accelerated SuperTrend calculation"""
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
    trailing_up_gpu = cuda.device_array(n, dtype=np.float64)
    trailing_dn_gpu = cuda.device_array(n, dtype=np.float64)

    # Configure CUDA kernel
    blocks_per_grid = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    # Launch CUDA kernel
    calculate_supertrend_cuda_kernel[blocks_per_grid, THREADS_PER_BLOCK](
        high_gpu, low_gpu, close_gpu, atr_length, factor, buffer_multiplier,
        up_gpu, dn_gpu, trend_gpu, trailing_up_gpu, trailing_dn_gpu
    )

    # Copy results back from GPU
    up = up_gpu.copy_to_host()
    dn = dn_gpu.copy_to_host()
    trend = trend_gpu.copy_to_host()
    trailing_up = trailing_up_gpu.copy_to_host()
    trailing_dn = trailing_dn_gpu.copy_to_host()

    # Add results to DataFrame
    df['up'] = up
    df['dn'] = dn
    df['trend'] = trend
    df['trailing_up'] = trailing_up
    df['trailing_dn'] = trailing_dn

    # Generate signals
    df['buy_signal'] = (df['trend'] == 1) & (df['low'] < df['trailing_up']) & (df['close'] > df['trailing_up'])
    df['sell_signal'] = df['trend'] == -1
    df['short_signal'] = (df['trend'] == -1) & (df['high'] > df['trailing_dn']) & (df['close'] < df['trailing_dn'])
    df['cover_signal'] = df['trend'] == 1

    # Clean up GPU memory
    try:
        del high_gpu, low_gpu, close_gpu, up_gpu, dn_gpu, trend_gpu, trailing_up_gpu, trailing_dn_gpu
        cuda.current_context().deallocations.clear()
    except Exception as e:
        print(f"Warning: Could not clear GPU memory: {e}", end='\r')

    return df

    
    
# Update the timestamp constant
CURRENT_UTC = "2025-05-03 04:02:45"
CURRENT_USER = "arullr001"

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

def calculate_supertrend(df, period, multiplier, price_range_1):
    """Wrapper function that chooses between CPU or GPU implementation"""
    try:
        if cuda.is_available():
            try:
                # Get GPU device
                device = cuda.get_current_device()
                
                # Try to get memory info using different methods
                try:
                    free_mem = device.mem_info()[0]  # First try this method
                except AttributeError:
                    try:
                        free_mem = device.memory_info().free  # Then try this
                    except AttributeError:
                        try:
                            ctx = cuda.current_context()
                            free_mem = ctx.get_memory_info().free
                        except AttributeError:
                            # If we can't get memory info, assume we don't have enough
                            print("Unable to determine GPU memory, falling back to CPU", end='\r')
                            return calculate_supertrend_cpu(df, period, multiplier, price_range_1)

                # Calculate required memory
                data_size = len(df) * 8 * 5  # Approximate size in bytes
                
                # Check if we have enough GPU memory (with buffer)
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

def cleanup_gpu_memory():
    """Clean up GPU memory to prevent leaks"""
    try:
        if cuda.is_available():
            import gc
            gc.collect()
            cuda.current_context().deallocations.clear()
            print("GPU memory cleaned", end='\r')
    except Exception as e:
        print(f"GPU memory cleanup error: {e}", end='\r')

def backtest_supertrend(df, atr_length, factor, buffer_multiplier, hard_stop_distance):
    """Optimized backtesting of the Supertrend strategy with hard stops"""
    # Calculate Supertrend and signals
    st_df = calculate_supertrend(df, atr_length, factor, buffer_multiplier)

    # Initialize trade tracking variables
    trades = []
    trade_number = 0
    in_long = False
    in_short = False
    entry_price = 0
    entry_time = None
    
    # Performance tracking
    consecutive_wins = 0
    current_streak = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    gross_profit = 0
    gross_loss = 0

    # Convert to numpy arrays for faster operations
    close = st_df['close'].values
    high = st_df['high'].values
    low = st_df['low'].values
    buy_signals = st_df['buy_signal'].values
    sell_signals = st_df['sell_signal'].values
    short_signals = st_df['short_signal'].values
    cover_signals = st_df['cover_signal'].values
    dates = st_df.index.to_numpy()
    supertrend = st_df['up'].values  # Assuming 'up' contains the Supertrend value

    # Loop through data for backtesting
    for i in range(1, len(st_df)):
        current_price = close[i]
        current_time = dates[i]

        # Handle existing long position
        if in_long:
            exit_condition = False
            exit_price = current_price
            exit_type = "trend_flip"

            # Check hard stop first
            if low[i] <= (supertrend[i] - hard_stop_distance):
                exit_condition = True
                exit_price = supertrend[i] - hard_stop_distance
                exit_type = "hard_stop"
            # Then check trend flip
            elif sell_signals[i]:
                exit_condition = True
                exit_type = "trend_flip"

            if exit_condition:
                points = exit_price - entry_price
                if points > 0:
                    gross_profit += points
                    current_streak = max(1, current_streak + 1)
                    max_consecutive_wins = max(max_consecutive_wins, current_streak)
                else:
                    gross_loss += abs(points)
                    current_streak = min(-1, current_streak - 1)
                    max_consecutive_losses = max(max_consecutive_losses, abs(current_streak))

                trades.append({
                    'trade_number': trade_number,
                    'direction': 'long',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': current_time,
                    'exit_price': exit_price,
                    'exit_type': exit_type,
                    'points': points
                })
                in_long = False

        # Handle existing short position
        if in_short:
            exit_condition = False
            exit_price = current_price
            exit_type = "trend_flip"

            # Check hard stop first
            if high[i] >= (supertrend[i] + hard_stop_distance):
                exit_condition = True
                exit_price = supertrend[i] + hard_stop_distance
                exit_type = "hard_stop"
            # Then check trend flip
            elif cover_signals[i]:
                exit_condition = True
                exit_type = "trend_flip"

            if exit_condition:
                points = entry_price - exit_price
                if points > 0:
                    gross_profit += points
                    current_streak = max(1, current_streak + 1)
                    max_consecutive_wins = max(max_consecutive_wins, current_streak)
                else:
                    gross_loss += abs(points)
                    current_streak = min(-1, current_streak - 1)
                    max_consecutive_losses = max(max_consecutive_losses, abs(current_streak))

                trades.append({
                    'trade_number': trade_number,
                    'direction': 'short',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': current_time,
                    'exit_price': exit_price,
                    'exit_type': exit_type,
                    'points': points
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
        points = close[-1] - entry_price
        if points > 0:
            gross_profit += points
        else:
            gross_loss += abs(points)
        trades.append({
            'trade_number': trade_number,
            'direction': 'long',
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': dates[-1],
            'exit_price': close[-1],
            'exit_type': 'end_of_data',
            'points': points
        })

    if in_short:
        points = entry_price - close[-1]
        if points > 0:
            gross_profit += points
        else:
            gross_loss += abs(points)
        trades.append({
            'trade_number': trade_number,
            'direction': 'short',
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': dates[-1],
            'exit_price': close[-1],
            'exit_type': 'end_of_data',
            'points': points
        })

    # Calculate performance metrics
    trade_count = len(trades)
    if trade_count > 0:
        winning_trades = len([t for t in trades if t['points'] > 0])
        avg_trade_duration = sum((t['exit_time'] - t['entry_time']).total_seconds() for t in trades) / trade_count
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        risk_adjusted_return = (gross_profit / trade_count) / (abs(gross_loss) / trade_count) if gross_loss != 0 else float('inf')
    else:
        winning_trades = 0
        avg_trade_duration = 0
        profit_factor = 0
        risk_adjusted_return = 0

    return {
        'parameters': {
            'atr_length': atr_length,
            'factor': factor,
            'buffer_multiplier': buffer_multiplier,
            'hard_stop_distance': hard_stop_distance
        },
        'total_profit': sum(trade['points'] for trade in trades) if trades else 0,
        'trade_count': trade_count,
        'win_rate': winning_trades / trade_count if trade_count > 0 else 0,
        'avg_trade_duration': avg_trade_duration,
        'profit_factor': profit_factor,
        'risk_adjusted_return': risk_adjusted_return,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'trades': trades
    }    
# Update the timestamp constant
CURRENT_UTC = "2025-05-03 04:03:37"  # Updated with the latest timestamp
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

def cleanup_memory():
    """Performs memory cleanup operations"""
    try:
        # Clear Python memory
        import gc
        gc.collect()

        # Clear GPU memory if available
        if cuda.is_available():
            cleanup_gpu_memory()

    except Exception as e:
        logging.getLogger('system_errors').error(f"Memory cleanup error: {str(e)}")

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

    def merge_batch_results(self):
        """Merges all batch results into final results"""
        all_results = []
        batch_files = glob.glob(os.path.join(self.dir_manager.csv_dumps_dir, 'batch_*.csv'))
        
        try:
            for file in batch_files:
                try:
                    df = pd.read_csv(file)
                    all_results.append(df)
                except Exception as e:
                    self.processing_logger.error(f"Error reading batch file {file}: {str(e)}")

            if all_results:
                final_df = pd.concat(all_results, ignore_index=True)
                final_df.sort_values('total_profit', ascending=False, inplace=True)
                
                # Save final results
                final_results_file = os.path.join(
                    self.dir_manager.final_results_dir, 
                    'best_combinations.csv'
                )
                final_df.to_csv(final_results_file, index=False)
                
                # Save final metadata
                final_metadata = {
                    'processed_at': self.current_utc,
                    'processed_by': self.user,
                    'total_combinations': len(final_df),
                    'top_profit': float(final_df['total_profit'].max()),
                    'avg_profit': float(final_df['total_profit'].mean()),
                    'total_batches': len(batch_files)
                }
                
                with open(os.path.join(
                    self.dir_manager.final_results_dir, 
                    'final_metadata.json'
                ), 'w') as f:
                    json.dump(final_metadata, f, indent=4)
                
                return final_df
                
            return pd.DataFrame()

        except Exception as e:
            self.system_logger.error(f"Error merging batch results: {str(e)}")
            return pd.DataFrame()


# Update the timestamp constant
CURRENT_UTC = "2025-05-03 04:11:02"
CURRENT_USER = "arullr001"


class ResultsManager:
    """Manages the generation and storage of results and analysis"""
    def __init__(self, directory_manager):
        self.dir_manager = directory_manager
        self.processing_logger = logging.getLogger('processing_errors')
        self.system_logger = logging.getLogger('system_errors')
        self.cached_results = {}

    def create_performance_summary(self, final_results_df):
        """Creates and saves a performance summary of the results"""
        try:
            if final_results_df.empty:
                print("\nNo results to summarize.")
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
                'timestamp': CURRENT_UTC,
                'generated_by': CURRENT_USER
            }

            # Save summary to JSON
            summary_file = os.path.join(self.dir_manager.final_results_dir, 'performance_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4)
            print(f"\nPerformance summary saved to: {summary_file}")

        except Exception as e:
            self.processing_logger.error(f"Error creating performance summary: {str(e)}")
            print(f"\nError creating performance summary: {str(e)}")

    def save_top_combinations_trades(self, final_results_df, original_data, top_n=5):
        """Saves detailed trade information for top performing parameter combinations"""
        try:
            if final_results_df.empty:
                print("\nNo results to save detailed trades.")
                return

            top_combinations = final_results_df.head(top_n)
            detailed_results_dir = os.path.join(self.dir_manager.final_results_dir, 'detailed_results')
            os.makedirs(detailed_results_dir, exist_ok=True)

            print(f"\nSaving detailed trade information for top {top_n} combinations...")
            for idx, row in top_combinations.iterrows():
                try:
                    print(f"\nProcessing rank {idx + 1}...")
                    result = backtest_supertrend(
                        original_data,
                        atr_length=row['atr_length'],
                        factor=row['factor'],
                        buffer_multiplier=row['buffer_multiplier'],
                        hard_stop_distance=row['hard_stop_distance']
                    )

                    # Save trades to CSV
                    trades_df = pd.DataFrame(result['trades'])
                    if not trades_df.empty:
                        trades_df['duration'] = pd.to_datetime(trades_df['exit_time']) - pd.to_datetime(trades_df['entry_time'])
                        
                        filename = f'trades_rank_{idx+1}_atr{int(row["atr_length"])}_f{row["factor"]:.2f}_b{row["buffer_multiplier"]:.2f}_s{int(row["hard_stop_distance"])}.csv'
                        trades_file = os.path.join(detailed_results_dir, filename)
                        trades_df.to_csv(trades_file, index=False)
                        print(f"Saved trades to: {filename}")

                        # Calculate and save statistics
                        stats = {
                            'rank': idx + 1,
                            'parameters': {
                                'atr_length': int(row['atr_length']),
                                'factor': float(row['factor']),
                                'buffer_multiplier': float(row['buffer_multiplier']),
                                'hard_stop_distance': int(row['hard_stop_distance'])
                            },
                            'performance_metrics': {
                                'total_trades': result['trade_count'],
                                'total_profit': result['total_profit'],
                                'win_rate': f"{result['win_rate']:.2%}",
                                'profit_factor': f"{result['profit_factor']:.2f}",
                                'risk_adjusted_return': f"{result['risk_adjusted_return']:.2f}",
                                'max_consecutive_wins': result['max_consecutive_wins'],
                                'max_consecutive_losses': result['max_consecutive_losses']
                            },
                            'generated_at': CURRENT_UTC,
                            'generated_by': CURRENT_USER
                        }

                        stats_filename = f'stats_rank_{idx+1}_atr{int(row["atr_length"])}_f{row["factor"]:.2f}_b{row["buffer_multiplier"]:.2f}_s{int(row["hard_stop_distance"])}.json'
                        stats_file = os.path.join(detailed_results_dir, stats_filename)
                        with open(stats_file, 'w') as f:
                            json.dump(stats, f, indent=4)
                        print(f"Saved statistics to: {stats_filename}")

                except Exception as e:
                    self.processing_logger.error(
                        f"Error processing detailed results for combination {row.to_dict()}: {str(e)}"
                    )
                    print(f"Error processing rank {idx + 1}: {str(e)}")

        except Exception as e:
            self.processing_logger.error(f"Error saving top combinations trades: {str(e)}")
            print(f"\nError saving top combinations trades: {str(e)}")

def create_performance_visualizations(self, df):
    """Creates visualization plots for the results"""
    try:
        # Use a default style instead of seaborn
        plt.style.use('default')  # Changed from 'seaborn' to 'default'

        # Create figures directory
        figures_dir = os.path.join(self.dir_manager.final_results_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)

        # 1. Profit Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df['total_profit'], bins=50, edgecolor='black')
        plt.title('Distribution of Total Profit')
        plt.xlabel('Total Profit')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(figures_dir, 'profit_distribution.png'))
        plt.close()

        # 2. Win Rate vs Profit
        plt.figure(figsize=(10, 6))
        plt.scatter(df['win_rate'], df['total_profit'], alpha=0.5)
        plt.title('Win Rate vs Total Profit')
        plt.xlabel('Win Rate')
        plt.ylabel('Total Profit')
        plt.savefig(os.path.join(figures_dir, 'winrate_vs_profit.png'))
        plt.close()

        # 3. Parameter Impact on Profit
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].scatter(df['period'], df['total_profit'], alpha=0.5)
        axes[0].set_title('Period vs Profit')
        axes[0].set_xlabel('Period')
        axes[0].set_ylabel('Total Profit')

        axes[1].scatter(df['multiplier'], df['total_profit'], alpha=0.5)
        axes[1].set_title('Multiplier vs Profit')
        axes[1].set_xlabel('Multiplier')
        axes[1].set_ylabel('Total Profit')

        axes[2].scatter(df['price_range_1'], df['total_profit'], alpha=0.5)
        axes[2].set_title('Price Range vs Profit')
        axes[2].set_xlabel('Price Range')
        axes[2].set_ylabel('Total Profit')

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'parameter_impact.png'))
        plt.close()

    except Exception as e:
        self.processing_logger.error(f"Error creating visualizations: {str(e)}")



def save_empty_results_file(file_path, results_dir, params, current_utc, current_user, total_combinations, reason="No Results"):
    """
    Creates empty results files with explanation when no results are generated
    Args:
        file_path (str): Path of the processed file
        results_dir (str): Directory to save the empty results
        params (dict or tuple): Parameter ranges or combination
        current_utc (str): Current UTC timestamp
        current_user (str): Current user's login
        total_combinations (int): Total number of combinations tested
        reason (str): Reason for no results (default "No Results")
    """
    # Handle both dictionary and tuple parameter inputs
    if isinstance(params, tuple):
        param_dict = {
            'atr_lengths': [params[0]],
            'factors': [params[1]],
            'buffers': [params[2]],
            'stops': [params[3]]
        }
    else:
        param_dict = params

    # Create the explanation message
    no_results_message = (
        f"{reason}\n"
        f"{'=' * len(reason)}\n"
        f"Processing Date: {current_utc}\n"
        f"User: {current_user}\n"
        f"File Processed: {os.path.basename(file_path)}\n"
        "\nParameter Ranges Tested:\n"
        f"- ATR Length: {min(param_dict['atr_lengths'])} to {max(param_dict['atr_lengths'])}\n"
        f"- Factor: {min(param_dict['factors'])} to {max(param_dict['factors'])}\n"
        f"- Buffer: {min(param_dict['buffers'])} to {max(param_dict['buffers'])}\n"
        f"- Hard Stop: {min(param_dict['stops'])} to {max(param_dict['stops'])}\n"
        f"\nTotal Combinations Tested: {total_combinations}\n"
        "\nReason: " + reason +
        "\nSuggestion: " +
        ("Try adjusting parameter ranges or reviewing data quality." if reason == "No parameter combinations generated profitable trades."
         else "Consider reviewing data quality or expanding parameter ranges.")
    )

    # Save as TXT
    no_results_txt = os.path.join(results_dir, 'no_results.txt')
    with open(no_results_txt, 'w') as f:
        f.write(no_results_message)

    # Save as CSV (empty with header)
    no_results_csv = os.path.join(results_dir, 'all_results.csv')
    empty_df = pd.DataFrame(columns=[
        'atr_length', 'factor', 'buffer_multiplier', 'hard_stop_distance',
        'total_profit', 'trade_count', 'win_rate', 'profit_factor',
        'risk_adjusted_return', 'max_consecutive_wins', 'max_consecutive_losses'
    ])
    empty_df.to_csv(no_results_csv, index=False)

    print(f"\n{reason}")
    print(f"Empty results files created in: {results_dir}")
    print("Check 'no_results.txt' for details.")



class GPUManager:
    """Manages GPU operations and distribution"""
    def __init__(self):
        self.logger = logging.getLogger('system_errors')
        self.available_gpus = self._get_available_gpus()
        print(f"Found {len(self.available_gpus)} available GPU(s)")
        
    def _get_available_gpus(self):
        """Get list of available CUDA GPUs"""
        try:
            gpu_list = []
            if cuda.is_available():
                for gpu_id in range(cuda.cuda.device_count()):
                    try:
                        cuda.select_device(gpu_id)
                        device = cuda.get_current_device()
                        try:
                            free_mem = device.mem_info()[0]
                        except:
                            ctx = cuda.current_context()
                            free_mem = ctx.get_memory_info().free

                        gpu = {
                            'id': gpu_id,
                            'name': device.name,
                            'free_memory': free_mem
                        }
                        gpu_list.append(gpu)
                        print(f"GPU {gpu_id}: {gpu['name']}")
                    except Exception as e:
                        self.logger.error(f"Error accessing GPU {gpu_id}: {str(e)}")
                        continue
            return gpu_list
        except Exception as e:
            self.logger.error(f"Error getting GPU list: {str(e)}")
            return []

    def distribute_work(self, param_combinations):
        """Distribute parameter combinations across available GPUs"""
        if not self.available_gpus:
            return [param_combinations]
        
        n_gpus = len(self.available_gpus)
        chunk_size = len(param_combinations) // n_gpus
        chunks = []
        
        for i in range(n_gpus):
            start = i * chunk_size
            end = start + chunk_size if i < n_gpus - 1 else len(param_combinations)
            chunks.append(param_combinations[start:end])
        
        return chunks

class ProgressTracker:
    """Tracks and displays progress of parameter combinations testing"""
    def __init__(self, total_combinations, dir_manager):
        self.total_combinations = total_combinations
        self.current_combination = 0
        self.start_time = time.time()
        self.dir_manager = dir_manager
        self.progress_file = os.path.join(dir_manager.base_dir, 'progress.json')
        self.initialize_progress_file()

    def initialize_progress_file(self):
        progress_data = {
            'total_combinations': self.total_combinations,
            'current_combination': 0,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'initialized'
        }
        self.save_progress(progress_data)

    def update(self, params, result=None):
        self.current_combination += 1
        elapsed_time = time.time() - self.start_time
        avg_time_per_combo = elapsed_time / self.current_combination
        remaining_combos = self.total_combinations - self.current_combination
        estimated_time_remaining = remaining_combos * avg_time_per_combo

        # Update progress display
        print('\r' + ' ' * 100 + '\r', end='')  # Clear line
        print(f"Progress: {self.current_combination}/{self.total_combinations} "
              f"({(self.current_combination/self.total_combinations)*100:.2f}%)")
        print(f"Testing combination {self.current_combination} of {self.total_combinations}")
        print(f"Parameters: ATR={params[0]}, Factor={params[1]}, "
              f"Buffer={params[2]}, Stop={params[3]}")
        
        if result:
            print(f"Results: Trades={result['trade_count']}, "
                  f"Profit={result['total_profit']:.2f}, "
                  f"Win Rate={result['win_rate']:.2%}")

        progress_data = {
            'total_combinations': self.total_combinations,
            'current_combination': self.current_combination,
            'elapsed_time': str(timedelta(seconds=int(elapsed_time))),
            'estimated_remaining': str(timedelta(seconds=int(estimated_time_remaining))),
            'current_parameters': {
                'atr_length': params[0],
                'factor': params[1],
                'buffer_multiplier': params[2],
                'hard_stop_distance': params[3]
            },
            'status': 'running'
        }
        
        if result:
            progress_data['latest_results'] = {
                'trade_count': result['trade_count'],
                'total_profit': result['total_profit'],
                'win_rate': result.get('win_rate', 0)
            }
        
        self.save_progress(progress_data)

    def save_progress(self, progress_data):
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=4)
        except Exception as e:
            print(f"Error saving progress: {str(e)}")


def main():
    """
    Main function to run the optimized Supertrend strategy backtester
    """
    try:
        CURRENT_UTC = "2025-06-12 16:32:59"  # Updated timestamp
        CURRENT_USER = "arullr001"           # Updated user

        start_time = time.time()
        print("=" * 50)
        print(" SUPER TREND STRATEGY BACKTESTER (OPTIMIZED) ".center(50, "="))
        print("=" * 50)
        print(f"Started at (UTC): {CURRENT_UTC}")
        print(f"User: {CURRENT_USER}")

        # Initialize managers
        dir_manager = DirectoryManager()
        mem_manager = MemoryManager()
        gpu_manager = GPUManager()  # New: Initialize GPU Manager
        batch_processor = BatchProcessor(dir_manager, mem_manager)
        results_manager = ResultsManager(dir_manager)

        # Create directory structure
        print(f"\nCreated working directory: {dir_manager.base_dir}")
        print("Directory structure:")
        print(f"├── csv_dumps/")
        print(f"├── error_logs/")
        print(f"└── final_results/")

        # Step 1: Find OHLC files
        data_dir = input("\nEnter the directory containing OHLC data files: ").strip()
        data_files = find_ohlc_files(data_dir)

        if not data_files:
            print("No data files found in the specified directory. Exiting.")
            sys.exit(1)

        # Step 2: Let user select files to process
        selected_files = select_files_to_process(data_files)
        if not selected_files:
            print("No files selected for processing. Exiting.")
            sys.exit(1)

        # Step 3: Get parameter inputs from user
        params = get_parameter_inputs()

        # Initialize progress tracker
        progress_tracker = ProgressTracker(params['total_combinations'], dir_manager)

        # Step 4: Process each file
        for file_path in selected_files:
            print(f"\nProcessing file: {os.path.basename(file_path)}")
            try:
                df = load_ohlc_data(file_path)
                print(f"\nLoaded data shape: {df.shape}")
                print(f"Date range: {df.index.min()} to {df.index.max()}")

                # Save input metadata
                input_metadata = {
                    'filename': os.path.basename(file_path),
                    'rows': len(df),
                    'date_range': {
                        'start': str(df.index.min()),
                        'end': str(df.index.max())
                    },
                    'processed_at': CURRENT_UTC,
                    'processed_by': CURRENT_USER,
                    'parameters_tested': {
                        'atr_lengths': f"{params['atr_lengths'][0]} to {params['atr_lengths'][-1]}",
                        'factors': f"{params['factors'][0]} to {params['factors'][-1]}",
                        'buffers': f"{params['buffers'][0]} to {params['buffers'][-1]}",
                        'stops': f"{params['stops'][0]} to {params['stops'][-1]}"
                    }
                }

                metadata_file = os.path.join(dir_manager.final_results_dir, 'input_metadata.json')
                with open(metadata_file, 'w') as f:
                    json.dump(input_metadata, f, indent=4)

                # Generate parameter combinations
                param_combinations = list(product(
                    params['atr_lengths'],
                    params['factors'],
                    params['buffers'],
                    params['stops']
                ))

                # New: Distribute work across available GPUs
                gpu_chunks = gpu_manager.distribute_work(param_combinations)
                total_combinations = len(param_combinations)

                all_results = []
                for chunk_idx, chunk in enumerate(gpu_chunks):
                    if gpu_manager.available_gpus:
                        gpu_id = chunk_idx % len(gpu_manager.available_gpus)
                        print(f"\nProcessing chunk {chunk_idx + 1} on GPU {gpu_id}")
                        cuda.select_device(gpu_id)
                    
                    batch_size = mem_manager.calculate_optimal_batch_size(
                        df.memory_usage().sum(),
                        len(chunk)
                    )

                    for batch_start in range(0, len(chunk), batch_size):
                        batch_end = min(batch_start + batch_size, len(chunk))
                        current_batch = chunk[batch_start:batch_end]

                        print(f"\nProcessing batch {batch_processor.current_batch + 1} "
                              f"({batch_start + 1}-{batch_end} of {len(chunk)})")
                        
                        try:
                            batch_results = []
                            for params in current_batch:
                                # Update progress tracker
                                progress_tracker.update(params)
                                
                                result = backtest_supertrend(
                                    df,
                                    atr_length=params[0],
                                    factor=params[1],
                                    buffer_multiplier=params[2],
                                    hard_stop_distance=params[3]
                                )

                                if result['trade_count'] > 0:
                                    batch_results.append(result)
                                    # Update progress with result
                                    progress_tracker.update(params, result)

                            if batch_results:
                                batch_data = [{
                                    'atr_length': r['parameters']['atr_length'],
                                    'factor': r['parameters']['factor'],
                                    'buffer_multiplier': r['parameters']['buffer_multiplier'],
                                    'hard_stop_distance': r['parameters']['hard_stop_distance'],
                                    'total_profit': r['total_profit'],
                                    'trade_count': r['trade_count'],
                                    'win_rate': r['win_rate'],
                                    'profit_factor': r['profit_factor'],
                                    'risk_adjusted_return': r['risk_adjusted_return'],
                                    'max_consecutive_wins': r['max_consecutive_wins'],
                                    'max_consecutive_losses': r['max_consecutive_losses']
                                } for r in batch_results]

                                batch_df = pd.DataFrame(batch_data)
                                batch_file = os.path.join(
                                    dir_manager.csv_dumps_dir,
                                    f'batch_{batch_processor.current_batch}.csv'
                                )
                                batch_df.to_csv(batch_file, index=False)
                                all_results.extend(batch_results)

                            batch_processor.current_batch += 1
                            cleanup_memory()

                        except Exception as e:
                            error_msg = f"Error processing batch: {str(e)}"
                            logging.getLogger('processing_errors').error(error_msg)
                            continue

                # Process and save final results
                if all_results:
                    results_data = [{
                        'atr_length': r['parameters']['atr_length'],
                        'factor': r['parameters']['factor'],
                        'buffer_multiplier': r['parameters']['buffer_multiplier'],
                        'hard_stop_distance': r['parameters']['hard_stop_distance'],
                        'total_profit': r['total_profit'],
                        'trade_count': r['trade_count'],
                        'win_rate': r['win_rate'],
                        'profit_factor': r['profit_factor'],
                        'risk_adjusted_return': r['risk_adjusted_return'],
                        'max_consecutive_wins': r['max_consecutive_wins'],
                        'max_consecutive_losses': r['max_consecutive_losses']
                    } for r in all_results]

                    final_results_df = pd.DataFrame(results_data)
                    final_results_df.sort_values('total_profit', ascending=False, inplace=True)

                    results_file = os.path.join(dir_manager.final_results_dir, 'all_results.csv')
                    final_results_df.to_csv(results_file, index=False)

                    results_manager.create_performance_summary(final_results_df)
                    results_manager.save_top_combinations_trades(final_results_df, df)

                else:
                    save_empty_results_file(
                        file_path, 
                        dir_manager.final_results_dir,
                        params,
                        CURRENT_UTC,
                        CURRENT_USER,
                        total_combinations
                    )

            except Exception as e:
                error_msg = f"Error processing file {file_path}: {str(e)}\n{traceback.format_exc()}"
                logging.getLogger('processing_errors').error(error_msg)
                continue

        print("\nProcessing complete!")
        print(f"Results directory: {dir_manager.base_dir}")
        print(f"Completed at (UTC): {CURRENT_UTC}")

    except Exception as e:
        error_msg = f"System error: {str(e)}\n{traceback.format_exc()}"
        logging.getLogger('system_errors').error(error_msg)
        print("\nA system error occurred. Check error logs for details.")
        sys.exit(1)

    finally:
        cleanup_memory()




# Make sure this follows immediately after the main() function
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.getLogger('system_errors').error(f"Fatal error: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)

    
