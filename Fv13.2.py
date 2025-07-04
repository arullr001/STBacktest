import os
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime, timedelta
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
import numba
from numba import cuda
try:
    import cupy as cp
except ImportError:
    cp = None
import concurrent.futures
import logging
from pathlib import Path
import math
import sys
import threading
import queue
import collections
import tkinter as tk
from tkinter import filedialog

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define constants
THREADS_PER_BLOCK = 256
MAX_RAM_USAGE_PERCENT = 90
BATCH_SIZE = 1000
CURRENT_UTC = "2025-05-03 03:57:20"  # Updated timestamp
CURRENT_USER = "arullr001"           # Updated user



# Utility Functions
def select_files_gui():
    """Open a file dialog for the user to select OHLC files (multiple selection allowed)"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames(
        title="Select OHLC Data Files (CSV, XLSX, XLS)",
        filetypes=[
            ("OHLC Data Files", "*.csv *.xlsx *.xls"),
            ("CSV Files", "*.csv"),
            ("Excel Files", "*.xlsx *.xls"),
            ("All Files", "*.*")
        ]
    )
    return list(file_paths)
    
    
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


def get_filtering_preferences():
    """Get user preferences for filtering criteria"""
    print("\n" + "=" * 50)
    print(" FILTERING PREFERENCES ".center(50, "="))
    print("=" * 50)
    
    use_filters = input("\nDo you want to use performance filters? (y/n): ").lower().strip() == 'y'
    
    if use_filters:
        use_drawdown = input("Use maximum drawdown filter? (y/n): ").lower().strip() == 'y'
        use_profit = input("Use minimum profit filter? (y/n): ").lower().strip() == 'y'
        use_min_trades = input("Use minimum trade count filter? (y/n): ").lower().strip() == 'y'
        
        filters = {
            'use_drawdown': use_drawdown,
            'use_profit': use_profit,
            'use_min_trades': use_min_trades,
            'max_drawdown': None,
            'min_profit': None,
            'min_trades': None
        }
        
        if use_drawdown:
            max_dd = input("Maximum allowed drawdown (as decimal, default 0.30 for 30%): ").strip()
            filters['max_drawdown'] = float(max_dd) if max_dd else 0.30
            
        if use_profit:
            min_profit = input("Minimum required profit (as decimal, default 0.15 for 15%): ").strip()
            filters['min_profit'] = float(min_profit) if min_profit else 0.15
            
        if use_min_trades:
            min_trades = input("Minimum required number of trades (default 30): ").strip()
            filters['min_trades'] = int(min_trades) if min_trades else 30
    else:
        filters = {
            'use_drawdown': False,
            'use_profit': False,
            'use_min_trades': True,  # Default to using min_trades filter
            'max_drawdown': None,
            'min_profit': None,
            'min_trades': 30  # Default minimum trade count
        }
    
    return filters


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



class HDDManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.batch_dir = os.path.join(base_dir, "batch_storage")
        self.temp_dir = os.path.join(base_dir, "temp_processing")
        self.buffer_size = 64 * 1024 * 1024  # 64MB buffer
        os.makedirs(self.batch_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

    def save_batch_to_disk(self, batch_data, batch_num):
        batch_file = os.path.join(self.batch_dir, f'batch_{batch_num}.npz')
        try:
            np.savez_compressed(batch_file, **batch_data)
            return True
        except Exception as e:
            logging.error(f"Error saving batch {batch_num}: {e}")
            return False

    def load_batch_from_disk(self, batch_num):
        batch_file = os.path.join(self.batch_dir, f'batch_{batch_num}.npz')
        try:
            return np.load(batch_file)
        except Exception as e:
            logging.error(f"Error loading batch {batch_num}: {e}")
            return None

    def cleanup_temp_files(self):
        for file in os.listdir(self.temp_dir):
            try:
                os.remove(os.path.join(self.temp_dir, file))
            except Exception as e:
                logging.error(f"Error cleaning temp file {file}: {e}")


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


# After your imports, add this block
# After imports section
def check_gpu_availability():
    """Check if CUDA GPU is available and initialize it"""
    try:
        if cuda.is_available():
            cuda.select_device(0)
            device = cuda.get_current_device()
            name = device.name
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            print(f"CUDA Device Available: {name}")
            return True
        else:
            print("No CUDA-capable GPU found, will use CPU implementation")
            return False
    except Exception as e:
        print(f"Error checking GPU availability: {str(e)}")
        print("Will use CPU implementation")
        return False

# Set the global variable correctly
HAS_GPU = check_gpu_availability()



def calculate_supertrend_cpu(df, atr_length, factor, buffer_multiplier):
    """CPU implementation of SuperTrend calculation matching Pinescript"""
    print("\nDebug - Starting CPU SuperTrend calculation")
    print(f"Parameters: ATR={atr_length}, Factor={factor}, Buffer={buffer_multiplier}")
    
    try:
        df = df.copy()
        
        # Calculate ATR and dynamic buffer
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.zeros(len(df))
        atr = np.zeros(len(df))
        
        # Calculate TR and ATR exactly as in Pinescript
        for i in range(len(df)):
            if i == 0:
                tr[i] = high[i] - low[i]
                atr[i] = tr[i]
            else:
                tr[i] = max(high[i] - low[i],
                           abs(high[i] - close[i-1]),
                           abs(low[i] - close[i-1]))
                atr[i] = (atr[i-1] * (atr_length - 1) + tr[i]) / atr_length
        
        # Calculate dynamic buffer as in Pinescript
        dynamic_buffer = atr * buffer_multiplier
        
        # Calculate basic bands
        hl2 = (high + low) / 2
        basic_upperband = hl2 + (factor * atr)
        basic_lowerband = hl2 - (factor * atr)
        
        # Initialize arrays
        final_upperband = np.zeros(len(df))
        final_lowerband = np.zeros(len(df))
        supertrend = np.zeros(len(df))
        direction = np.zeros(len(df))
        
        # Calculate Supertrend
        for i in range(1, len(df)):
            # Calculate upper band
            if basic_upperband[i] < final_upperband[i-1] or close[i-1] > final_upperband[i-1]:
                final_upperband[i] = basic_upperband[i]
            else:
                final_upperband[i] = final_upperband[i-1]
                
            # Calculate lower band
            if basic_lowerband[i] > final_lowerband[i-1] or close[i-1] < final_lowerband[i-1]:
                final_lowerband[i] = basic_lowerband[i]
            else:
                final_lowerband[i] = final_lowerband[i-1]
            
            # Determine trend direction
            if close[i] > final_upperband[i-1]:
                direction[i] = -1  # Uptrend (matching Pinescript's direction)
                supertrend[i] = final_lowerband[i]
            elif close[i] < final_lowerband[i-1]:
                direction[i] = 1   # Downtrend (matching Pinescript's direction)
                supertrend[i] = final_upperband[i]
            else:
                direction[i] = direction[i-1]
                supertrend[i] = supertrend[i-1]
        
        # Add results to DataFrame
        df['supertrend'] = supertrend
        df['direction'] = direction
        
        # Calculate buffer zones exactly as in Pinescript
        df['up_trend_buffer'] = np.where(direction < 0, 
                                        supertrend + dynamic_buffer,
                                        np.nan)
        df['down_trend_buffer'] = np.where(direction > 0,
                                          supertrend - dynamic_buffer,
                                          np.nan)
        
        # Generate signals using Pinescript logic
        df['buy_signal'] = (
            (df['direction'] < 0) &
            (df['close'] >= df['supertrend']) & 
            (df['close'] <= df['up_trend_buffer'])
        )
        
        df['sell_signal'] = (
            (df['direction'] > 0) & \
            (df['close'] < df['supertrend']) &
            (df['close'] >= df['down_trend_buffer'])
        )
        
        print("\nDebug - Signal Statistics:")
        print(f"Buy Signals: {df['buy_signal'].sum()} ({(df['buy_signal'].sum()/len(df))*100:.2f}%)")
        print(f"Sell Signals: {df['sell_signal'].sum()} ({(df['sell_signal'].sum()/len(df))*100:.2f}%)")
        
        return df
        
    except Exception as e:
        print(f"Error in CPU calculation: {str(e)}")
        raise



@cuda.jit
def calculate_supertrend_cuda_kernel(high, low, close, atr_length, factor, buffer_multiplier,
                                   supertrend, direction, up_trend_buffer, down_trend_buffer):
    """CUDA kernel for SuperTrend calculation matching Pinescript"""
    i = cuda.grid(1)
    if i >= len(close):
        return

    # Calculate ATR
    if i == 0:
        tr = high[0] - low[0]
        atr = tr
    else:
        tr = max(high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1]))
        atr = (atr * (atr_length - 1) + tr) / atr_length

    # Calculate dynamic buffer
    dynamic_buffer = atr * buffer_multiplier
    
    # Calculate basic bands
    hl2 = (high[i] + low[i]) / 2
    basic_upperband = hl2 + (factor * atr)
    basic_lowerband = hl2 - (factor * atr)

    if i == 0:
        supertrend[i] = hl2
        direction[i] = 0
    else:
        # Determine trend direction and supertrend value exactly as in Pinescript
        if close[i] > supertrend[i-1]:
            direction[i] = -1  # Uptrend
            supertrend[i] = basic_lowerband
        elif close[i] < supertrend[i-1]:
            direction[i] = 1   # Downtrend
            supertrend[i] = basic_upperband
        else:
            direction[i] = direction[i-1]
            supertrend[i] = supertrend[i-1]

    # Calculate buffer zones exactly as in Pinescript
    if direction[i] < 0:
        up_trend_buffer[i] = supertrend[i] + dynamic_buffer
        down_trend_buffer[i] = 0.0  # nan equivalent
    elif direction[i] > 0:
        up_trend_buffer[i] = 0.0    # nan equivalent
        down_trend_buffer[i] = supertrend[i] - dynamic_buffer


def calculate_supertrend_gpu(df, atr_length, factor, buffer_multiplier):
    """GPU-accelerated SuperTrend calculation matching Pinescript"""
    df = df.copy()
    n = len(df)

    # Prepare input arrays
    high = cuda.to_device(df['high'].values.astype(np.float64))
    low = cuda.to_device(df['low'].values.astype(np.float64))
    close = cuda.to_device(df['close'].values.astype(np.float64))

    # Create output arrays
    supertrend = cuda.device_array(n, dtype=np.float64)
    direction = cuda.device_array(n, dtype=np.float64)
    up_trend_buffer = cuda.device_array(n, dtype=np.float64)
    down_trend_buffer = cuda.device_array(n, dtype=np.float64)

    # Configure and launch kernel
    threads_per_block = THREADS_PER_BLOCK
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    calculate_supertrend_cuda_kernel[blocks_per_grid, threads_per_block](
        high, low, close, atr_length, factor, buffer_multiplier,
        supertrend, direction, up_trend_buffer, down_trend_buffer
    )

    # Copy results back to host
    df['supertrend'] = supertrend.copy_to_host()
    df['direction'] = direction.copy_to_host()
    df['up_trend_buffer'] = up_trend_buffer.copy_to_host()
    df['down_trend_buffer'] = down_trend_buffer.copy_to_host()

    # Generate signals using Pinescript logic
    df['buy_signal'] = (
        (df['direction'] < 0) &
        (df['close'] >= df['supertrend']) & 
        (df['close'] <= df['up_trend_buffer'])
    )
        
    df['sell_signal'] = (
        (df['direction'] > 0) & \
        (df['close'] < df['supertrend']) &
        (df['close'] >= df['down_trend_buffer'])
    )
        
    return df


def calculate_supertrend(df, atr_length, factor, buffer_multiplier):
    """
    Wrapper function that chooses between CPU or GPU implementation with enhanced debugging
    """
    print(f"\nDebug - Starting SuperTrend calculation:")
    print(f"Parameters: ATR={atr_length}, Factor={factor}, Buffer={buffer_multiplier}")
    print(f"Input DataFrame shape: {df.shape}")
    
    try:
        if cuda.is_available():
            try:
                ctx = cuda.current_context()
                free_mem = ctx.get_memory_info().free

                data_size = len(df) * 8 * 5  # Approximate size in bytes
                print(f"Debug - GPU Memory Check:")
                print(f"Required memory: {data_size / (1024*1024):.2f} MB")
                print(f"Available GPU memory: {free_mem / (1024*1024):.2f} MB")
                
                if free_mem < 1024*1024*100:  # Minimum 100MB required
                    print("Insufficient GPU memory, using CPU")
                    return calculate_supertrend_cpu(df, atr_length, factor, buffer_multiplier)
                
                if free_mem > data_size * 3:
                    print("Using GPU acceleration")
                    result_df = calculate_supertrend_gpu(df, atr_length, factor, buffer_multiplier)
                else:
                    print("Insufficient GPU memory, using CPU")
                    result_df = calculate_supertrend_cpu(df, atr_length, factor, buffer_multiplier)
            except Exception as e:
                print(f"GPU initialization failed: {str(e)}")
                print("Falling back to CPU implementation")
                result_df = calculate_supertrend_cpu(df, atr_length, factor, buffer_multiplier)
        else:
            print("No GPU available, using CPU implementation")
            result_df = calculate_supertrend_cpu(df, atr_length, factor, buffer_multiplier)

        # Verify calculations
        print("\nDebug - Verification of calculations:")
        print(f"SuperTrend values generated: {all(col in result_df.columns for col in ['supertrend', 'direction'])}")
        print(f"Trend values present: {result_df['direction'].nunique()} unique trends")
        print(f"Buffer zones calculated: {all(col in result_df.columns for col in ['up_trend_buffer', 'down_trend_buffer'])}")
        
        # Signal generation check
        buy_signals = result_df['buy_signal'].sum()
        sell_signals = result_df['sell_signal'].sum()
        
        print("\nDebug - Signal Generation:")
        print(f"Buy signals: {buy_signals}")
        print(f"Sell signals: {sell_signals}")
        
        if buy_signals == 0 and sell_signals == 0:
            print("\nWarning: No entry signals generated!")
            print("Parameters may be too restrictive")
        
        # Basic signal validation
        signal_validation = {
            'has_entries': buy_signals > 0 or sell_signals > 0,
            'reasonable_signal_ratio': 0.0001 <= (buy_signals + sell_signals) / len(df) <= 0.25
        }
        
        print("\nDebug - Signal Validation:")
        for check, passed in signal_validation.items():
            print(f"{check}: {'Passed' if passed else 'Failed'}")
        
        # Final verification
        required_columns = ['open', 'high', 'low', 'close', 'supertrend', 'direction',
                          'up_trend_buffer', 'down_trend_buffer', 'buy_signal', 'sell_signal']
        
        missing_columns = [col for col in required_columns if col not in result_df.columns]
        if missing_columns:
            print(f"\nWarning: Missing columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        print("\nDebug - SuperTrend calculation completed successfully")
        return result_df

    except Exception as e:
        error_msg = f"Error in SuperTrend calculation: {str(e)}\n{traceback.format_exc()}"
        print(f"\nDebug - Error in calculate_supertrend:")
        print(error_msg)
        logging.getLogger('processing_errors').error(error_msg)
        raise



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


class StatusDisplay:
    """Tracks and displays progress of parameter combinations testing"""
    def __init__(self, params, total_combinations, filename, filters=None):
        # Create a separate window for the status display if in a GUI environment
        # For terminal, we'll use a simpler approach
        try:
            self.params = params
            self.total_combinations = total_combinations
            self.filename = filename
            self.current_combo = 0
            self.start_time = time.time()
            self.top_combinations = []
            self.running = True
            
            if filters is None:
                self.filters = {
                    'use_drawdown': False,
                    'use_profit': False,
                    'use_min_trades': False,
                    'max_drawdown': None,
                    'min_profit': None,
                    'min_trades': None
                }
            else:
                self.filters = filters
            
            # Print a separator that will make it easier to find the status
            print("\n" + "=" * 80)
            print(" STATUS DISPLAY - Refer to this line to find status updates ".center(80, "="))
            print("=" * 80 + "\n")
            
            # Initial display
            self._draw_box()
            
        except Exception as e:
            print(f"Error in StatusDisplay initialization: {str(e)}")
            print(traceback.format_exc())
            raise

    def _format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _draw_box(self):
        """Draw the status box with a clear separator for visibility"""
        try:
            # Start with a clear separator to find the status
            print("\n" + "=" * 80)
            print(" CURRENT STATUS ".center(80, "="))
            
            # Box contents
            print("=" * 70)
            print("║" + " SUPERTREND BACKTESTER STATUS ".center(68) + "║")
            print("║" + f" {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} ".center(68) + "║")
            print("=" * 70)

            # Progress stats
            print(f"Progress: {self.current_combo}/{self.total_combinations} " + 
                  f"({(self.current_combo/self.total_combinations)*100:.1f}%)")
            
            # Time info
            elapsed = time.time() - self.start_time
            elapsed_str = self._format_time(elapsed)
            print(f"Time Elapsed: {elapsed_str}")
            
            if self.current_combo > 0:
                remaining = (elapsed / self.current_combo) * (self.total_combinations - self.current_combo)
                eta_str = self._format_time(remaining)
                print(f"ETA: {eta_str}")
            
            # Top combinations
            print("\nTop Combinations (by Profit Factor):")
            if self.top_combinations:
                for i, combo in enumerate(self.top_combinations[-3:], 1):
                    print(f"{i}. ATR:{combo['atr']} F:{combo['factor']:.1f} " +
                          f"B:{combo['buffer']:.1f} S:{combo['stop']} | " +
                          f"PF:{combo['pf']:.2f} | WR:{combo['wr']:.1%} | Trades:{combo['trades']}")
            else:
                print("No combinations found yet")
                
            print("=" * 80)
            print(" END OF STATUS - Processing continues below ".center(80, "="))
            print("=" * 80 + "\n")
            
            sys.stdout.flush()
            
        except Exception as e:
            print(f"Error in _draw_box: {str(e)}")
            print(traceback.format_exc())

    def update(self, current_combo, top_combo=None):
        """Update status with current progress"""
        self.current_combo = current_combo
        if top_combo:
            self._update_top_combinations(top_combo)
        self._draw_box()

    def _update_top_combinations(self, combo):
        """Update list of top combinations"""
        trade_count = combo.get('trade_count', 0)
        min_trades = self.filters.get('min_trades', 0) if self.filters.get('use_min_trades', False) else 0
        
        if trade_count >= min_trades:
            new_combo = {
                'atr': combo['parameters']['atr_length'],
                'factor': combo['parameters']['factor'],
                'buffer': combo['parameters']['buffer_multiplier'],
                'stop': combo['parameters']['hard_stop_distance'],
                'pf': combo.get('profit_factor', 0),
                'wr': combo.get('win_rate', 0),
                'trades': trade_count
            }
            
            self.top_combinations.append(new_combo)
            self.top_combinations.sort(key=lambda x: x['pf'], reverse=True)
            self.top_combinations = self.top_combinations[:3]

    def cleanup(self):
        self.running = False
        print("\n" + "=" * 80)
        print(" STATUS DISPLAY TERMINATED ".center(80, "="))
        print("=" * 80 + "\n")


def backtest_supertrend(df, parameters):
    """
    Backtest SuperTrend strategy with specified parameters
    
    Args:
        df: DataFrame with OHLCV data
        parameters: Dict with strategy parameters
    
    Returns:
        Dict with backtest results
    """
    # Extract parameters
    atr_length = parameters['atr_length']
    factor = parameters['factor']
    buffer_multiplier = parameters['buffer_multiplier'] 
    hard_stop_distance = parameters['hard_stop_distance']
    
    # Calculate SuperTrend
    df_tmp = df.copy()
    try:
        df_tmp = calculate_supertrend(df_tmp, atr_length, factor, buffer_multiplier)
    except Exception as e:
        print(f"Error calculating SuperTrend: {e}")
        return None
    
    # Debug info - critical for diagnosing
    print(f"Debug - SuperTrend calculation result:")
    print(f"Data shape after SuperTrend: {df_tmp.shape}")
    print(f"Buy signals count: {df_tmp['buy_signal'].sum()}")
    print(f"Sell signals count: {df_tmp['sell_signal'].sum()}")
    
    # Prepare for backtesting
    dates = df_tmp.index.values
    opens = df_tmp['open'].values
    highs = df_tmp['high'].values
    lows = df_tmp['low'].values
    closes = df_tmp['close'].values
    directions = df_tmp['direction'].values  # Using direction instead of trend
    
    # Track trades
    trades = []
    trade_number = 0
    
    # Position tracking
    in_long = False
    in_short = False
    entry_price = 0
    entry_time = None
    high_since_entry = 0
    low_since_entry = float('inf')
    
    # Trading loop - COMPLETE IMPLEMENTATION
    for i in range(1, len(df_tmp)):
        current_time = dates[i]
        current_price = closes[i]
        previous_direction = directions[i-1]
        current_direction = directions[i]
        
        # Check for long exit
        if in_long:
            exit_condition = False
            exit_price = current_price
            exit_type = "trend_flip"
            
            # Update tracking values
            high_since_entry = max(high_since_entry, highs[i])
            
            # Check for time-based exit (after 2 days)
            entry_datetime = pd.to_datetime(entry_time)
            current_datetime = pd.to_datetime(current_time)
            days_in_trade = (current_datetime - entry_datetime).total_seconds() / (24 * 60 * 60)
            
            if days_in_trade >= 2:
                exit_condition = True
                exit_type = "time_based_exit"
                
            # Check other exit conditions
            elif current_direction > 0 and previous_direction < 0:  # Direction flipped to bearish
                exit_condition = True
                exit_type = "trend_flip"
                
            elif (entry_price - current_price) >= hard_stop_distance:
                # Hard stop loss hit
                exit_condition = True
                exit_price = entry_price - hard_stop_distance
                exit_type = "stop_loss"
                
            # Process exit if needed
            if exit_condition:
                points = exit_price - entry_price
                pnl_pct = points / entry_price
                
                # Calculate duration using candle count
                entry_idx = np.where(dates == entry_time)[0][0]
                exit_idx = i
                candle_count = exit_idx - entry_idx
                
                # Each candle is 5 minutes (assuming 5-min chart)
                trade_duration = candle_count * (5/60)  # Convert to hours
                
                trades.append({
                    'trade_number': trade_number,
                    'direction': 'long',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': current_time, 
                    'exit_price': exit_price,
                    'exit_type': exit_type,
                    'points': points,
                    'pnl_percent': pnl_pct,
                    'duration': trade_duration,
                    'candle_count': candle_count
                })
                
                in_long = False
                high_since_entry = 0
                
        # Check for short exit
        elif in_short:
            exit_condition = False
            exit_price = current_price
            exit_type = "trend_flip"
            
            # Update tracking values
            low_since_entry = min(low_since_entry, lows[i])
            
            # Check for time-based exit (after 2 days)
            entry_datetime = pd.to_datetime(entry_time)
            current_datetime = pd.to_datetime(current_time)
            days_in_trade = (current_datetime - entry_datetime).total_seconds() / (24 * 60 * 60)
            
            if days_in_trade >= 2:
                exit_condition = True
                exit_type = "time_based_exit"
                
            # Check other exit conditions
            elif current_direction < 0 and previous_direction > 0:
                # Direction flipped to bullish - exit short
                exit_condition = True
                exit_type = "trend_flip"
                
            elif (current_price - entry_price) >= hard_stop_distance:
                # Hard stop loss hit
                exit_condition = True
                exit_price = entry_price + hard_stop_distance
                exit_type = "stop_loss"
                
            # Process exit if needed
            if exit_condition:
                points = entry_price - exit_price
                pnl_pct = points / entry_price
                
                # Calculate duration using candle count
                entry_idx = np.where(dates == entry_time)[0][0]
                exit_idx = i
                candle_count = exit_idx - entry_idx
                
                # Each candle is 5 minutes (assuming 5-min chart)
                trade_duration = candle_count * (5/60)  # Convert to hours
                
                trades.append({
                    'trade_number': trade_number,
                    'direction': 'short',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': current_time,
                    'exit_price': exit_price,
                    'exit_type': exit_type,
                    'points': points,
                    'pnl_percent': pnl_pct,
                    'duration': trade_duration,
                    'candle_count': candle_count
                })
                
                in_short = False
                low_since_entry = float('inf')
        
        # Check for entries (only if not in a position)
        if not in_long and not in_short:
            # Use buy_signal and sell_signal from SuperTrend calculation
            if df_tmp['buy_signal'].iloc[i]:  # Using the buy_signal column from SuperTrend
                trade_number += 1
                entry_price = current_price
                entry_time = current_time
                high_since_entry = highs[i]
                in_long = True
                print(f"Debug: Long entry at index {i}, price {entry_price}, time {entry_time}")
                
            elif df_tmp['sell_signal'].iloc[i]:  # Using the sell_signal column from SuperTrend
                trade_number += 1
                entry_price = current_price
                entry_time = current_time
                low_since_entry = lows[i]
                in_short = True
                print(f"Debug: Short entry at index {i}, price {entry_price}, time {entry_time}")
    
    # Close any open position at the end
    if in_long:
        points = closes[-1] - entry_price
        pnl_pct = points / entry_price
        
        entry_idx = np.where(dates == entry_time)[0][0]
        exit_idx = len(df_tmp) - 1
        candle_count = exit_idx - entry_idx
        trade_duration = candle_count * (5/60)  # Convert to hours
        
        trades.append({
            'trade_number': trade_number,
            'direction': 'long',
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': dates[-1],
            'exit_price': closes[-1],
            'exit_type': 'end_of_data',
            'points': points,
            'pnl_percent': pnl_pct,
            'duration': trade_duration,
            'candle_count': candle_count
        })
        
    elif in_short:
        points = entry_price - closes[-1]
        pnl_pct = points / entry_price
        
        entry_idx = np.where(dates == entry_time)[0][0]
        exit_idx = len(df_tmp) - 1
        candle_count = exit_idx - entry_idx
        trade_duration = candle_count * (5/60)  # Convert to hours
        
        trades.append({
            'trade_number': trade_number,
            'direction': 'short',
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': dates[-1],
            'exit_price': closes[-1],
            'exit_type': 'end_of_data',
            'points': points,
            'pnl_percent': pnl_pct,
            'duration': trade_duration,
            'candle_count': candle_count
        })
    
    # Calculate backtest statistics
    trade_count = len(trades)
    print(f"Debug: Total trades generated: {trade_count}")
    
    if trade_count > 0:
        win_count = sum(1 for t in trades if t['points'] > 0)
        loss_count = sum(1 for t in trades if t['points'] <= 0)
        win_rate = win_count / trade_count if trade_count > 0 else 0
        
        gross_profit = sum(t['points'] for t in trades if t['points'] > 0)
        gross_loss = sum(t['points'] for t in trades if t['points'] <= 0)
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        avg_profit = gross_profit / win_count if win_count > 0 else 0
        avg_loss = gross_loss / loss_count if loss_count > 0 else 0
        
        avg_trade_duration = sum(t['duration'] for t in trades) / trade_count
        avg_candle_count = sum(t['candle_count'] for t in trades) / trade_count
        
        # Calculate drawdown
        equity_curve = [0]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade['points'])
        
        peak = 0
        drawdown = 0
        max_drawdown = 0
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            
            drawdown = peak - equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calculate additional metrics
        returns = [t['points'] for t in trades]
        returns_array = np.array(returns)
        sharpe_ratio = returns_array.mean() / returns_array.std() if returns_array.std() != 0 else 0
        
        # Calculate max consecutive wins/losses
        win_loss_array = np.array([1 if t['points'] > 0 else 0 for t in trades])
        max_consecutive_wins = max_consecutive(win_loss_array, 1)
        max_consecutive_losses = max_consecutive(win_loss_array, 0)
        
        # Total profit
        total_profit = sum(t['points'] for t in trades)
        
        # Prepare results
        results = {
            'parameters': parameters,
            'trade_count': trade_count,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'avg_trade_duration': avg_trade_duration,
            'avg_candle_count': avg_candle_count,
            'sharpe_ratio': sharpe_ratio,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'trades': trades
        }
    else:
        results = {
            'parameters': parameters,
            'trade_count': 0,
            'win_count': 0,
            'loss_count': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'total_profit': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'max_drawdown': 0,
            'avg_trade_duration': 0,
            'avg_candle_count': 0,
            'sharpe_ratio': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'trades': []
        }
    
    return results

# Helper function for calculating max consecutive wins/losses
def max_consecutive(arr, val):
    """Calculate maximum consecutive occurrences of val in arr"""
    max_count = 0
    current_count = 0
    
    for v in arr:
        if v == val:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
            
    return max_count


def rank_parameter_combinations(self, final_results_df, filters):
    """Rank parameter combinations based on multiple metrics with optional filtering"""
    try:
        # DEBUG: Print columns to verify what's available
        print("\nDEBUG - Available columns in results dataframe:")
        print(list(final_results_df.columns))
        
        # Start with the full dataset
        filtered_df = final_results_df.copy()
        
        # SAFE CHECK: Add any missing columns with default values
        for col in ['avg_trade_duration', 'profit_factor', 'sharpe_ratio', 'win_rate', 'max_drawdown']:
            if col not in filtered_df.columns:
                print(f"Warning: '{col}' not found in results. Adding with default value.")
                filtered_df[col] = 0.0
        
        # Apply filters based on preferences
        if filters.get('use_drawdown', False) and filters.get('max_drawdown') is not None:
            filtered_df = filtered_df[
                filtered_df['max_drawdown'] <= filters['max_drawdown']
            ]
            print(f"Applied drawdown filter: {filters['max_drawdown']:.2%}")
        
        if filters.get('use_profit', False) and filters.get('min_profit') is not None:
            filtered_df = filtered_df[
                filtered_df['total_profit'] >= filters['min_profit']
            ]
            print(f"Applied profit filter: {filters['min_profit']:.2%}")

        if filtered_df.empty:
            print("\nNo parameter combinations meet the filtering criteria.")
            print("Consider adjusting the filtering thresholds or running without filters.")
            return pd.DataFrame()
        
        # Apply minimum trade count filter (if configured)
        min_trades = 30  # Default value
        if filters.get('use_min_trades', True) and filters.get('min_trades') is not None:
            min_trades = filters['min_trades']
        
        print(f"Combinations before trade filter: {len(filtered_df)}")
        filtered_df = filtered_df[filtered_df['trade_count'] >= min_trades]
        print(f"Combinations after trade filter: {len(filtered_df)}")

        if filtered_df.empty:
            print(f"\nNo parameter combinations meet the minimum trade count of {min_trades}.")
            print("Consider relaxing filter criteria or expanding parameter ranges.")
            return pd.DataFrame()

        # Calculate composite score
        filtered_df['composite_score'] = (
            filtered_df['profit_factor'] * 0.4 +    # 40% weight
            filtered_df['sharpe_ratio'] * 0.3 +     # 30% weight
            filtered_df['win_rate'] * 0.3           # 30% weight
        )

        # Sort by profit factor first, then use composite score for tie-breaking
        ranked_df = filtered_df.sort_values(
            ['profit_factor', 'composite_score', 'win_rate'],
            ascending=[False, False, False]
        )

        # Get top 5 combinations
        top_5 = ranked_df.head(5)

        # Save detailed results for top 5
        detailed_results_dir = os.path.join(self.dir_manager.final_results_dir, 'top_5_combinations')
        os.makedirs(detailed_results_dir, exist_ok=True)

        # Save results with all metrics
        results_file = os.path.join(detailed_results_dir, 'top_5_ranked_results.csv')
        top_5.to_csv(results_file, index=False)

        # Create detailed summary
        summary_data = {
            'analysis_timestamp': self.current_utc,
            'analyzed_by': self.user,
            'filtering_criteria': {
                'filters_used': bool(filters.get('use_drawdown', False) or filters.get('use_profit', False) or filters.get('use_min_trades', True)),
                'drawdown_filter': {
                    'used': filters.get('use_drawdown', False),
                    'threshold': filters.get('max_drawdown')
                },
                'profit_filter': {
                    'used': filters.get('use_profit', False),
                    'threshold': filters.get('min_profit')
                },
                'min_trades_filter': {
                    'used': filters.get('use_min_trades', True),
                    'threshold': min_trades
                }
            },
            'total_combinations_analyzed': len(final_results_df),
            'combinations_after_filtering': len(filtered_df),
            'top_5_combinations': []
        }

        print("\nTop 5 Parameter Combinations:")
        for idx, (_, row) in enumerate(top_5.iterrows(), 1):
            # Critical fix: Create metrics dictionary carefully to avoid KeyError
            metrics_dict = {
                'profit_factor': float(row.get('profit_factor', 0.0)),
                'sharpe_ratio': float(row.get('sharpe_ratio', 0.0)),
                'win_rate': float(row.get('win_rate', 0.0)),
                'max_drawdown': float(row.get('max_drawdown', 0.0)),
                'total_profit': float(row.get('total_profit', 0.0)),
                'expectancy': float(row.get('expectancy', 0.0)),
                'trade_count': int(row.get('trade_count', 0))
            }
            
            # Only add avg_trade_duration if it exists
            if 'avg_trade_duration' in row and not pd.isna(row['avg_trade_duration']):
                metrics_dict['avg_trade_duration'] = float(row['avg_trade_duration'])
            
            combo_data = {
                'rank': idx,
                'parameters': {
                    'atr_length': int(row.get('atr_length', 0)),
                    'factor': float(row.get('factor', 0.0)),
                    'buffer_multiplier': float(row.get('buffer_multiplier', 0.0)),
                    'hard_stop_distance': float(row.get('hard_stop_distance', 0.0))
                },
                'metrics': metrics_dict
            }
            
            summary_data['top_5_combinations'].append(combo_data)
        
            print(f"\nRank {idx}:")
            print(f"Parameters: ATR={row.get('atr_length', 0)}, Factor={row.get('factor', 0.0):.2f}, "
                  f"Buffer={row.get('buffer_multiplier', 0.0):.2f}, Stop={row.get('hard_stop_distance', 0.0)}")
            print(f"Profit Factor: {row.get('profit_factor', 0.0):.2f}")
            print(f"Sharpe Ratio: {row.get('sharpe_ratio', 0.0):.2f}")
            print(f"Win Rate: {row.get('win_rate', 0.0):.2%}")
            print(f"Max Drawdown: {row.get('max_drawdown', 0.0):.2%}")
            print(f"Net Profit: {row.get('total_profit', 0.0):.2f}")
            print(f"Total Trades: {int(row.get('trade_count', 0))}")
            
            # Only print avg_trade_duration if it exists and is not NaN
            if 'avg_trade_duration' in row and not pd.isna(row['avg_trade_duration']):
                print(f"Avg Duration: {row.get('avg_trade_duration', 0.0):.2f} hours")

        # Save summary to JSON
        summary_file = os.path.join(detailed_results_dir, 'top_5_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=4)

        return top_5

    except Exception as e:
        error_msg = f"Error ranking parameter combinations: {str(e)}"
        print(f"\n{error_msg}")
        print(traceback.format_exc())  # Print full stack trace for better debugging
        return pd.DataFrame()

def process_param_combo(args):
    """Process a single parameter combination with enhanced debugging"""
    try:
        df, atr_length, factor, buffer_multiplier, hard_stop_distance = args
        
        print(f"\nDebug - Processing combination:")
        print(f"ATR={atr_length}, Factor={factor}, Buffer={buffer_multiplier}, Stop={hard_stop_distance}")
        
        print("\nDebug - Data verification:")
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Create parameters dictionary instead of passing as kwargs
        parameters = {
            'atr_length': atr_length,
            'factor': factor,
            'buffer_multiplier': buffer_multiplier,
            'hard_stop_distance': hard_stop_distance
        }
        
        # Call with dictionary of parameters
        result = backtest_supertrend(df, parameters)
        
        if result:
            print("\nDebug - Result metrics:")
            print(f"Trade count: {result['trade_count']}")
            print(f"Total profit: {result['total_profit']:.2f}")
            if result['trade_count'] > 0:
                print(f"Win rate: {result['win_rate']:.2%}")
                print(f"Max drawdown: {result['max_drawdown']:.4f}")
                print(f"Sharpe ratio: {result['sharpe_ratio']:.4f}")
                print(f"Profit factor: {result['profit_factor']:.2f}")
                print(f"Avg trade duration: {result['avg_trade_duration']:.2f} hours")
                print(f"Max consecutive wins: {result['max_consecutive_wins']}")
                print(f"Max consecutive losses: {result['max_consecutive_losses']}")
            
            return result
        else:
            print("Warning: No valid result generated")
            return None
        
    except Exception as e:
        error_msg = f"Error in process_param_combo: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        logging.getLogger('processing_errors').error(error_msg)
        return None



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
    def __init__(self, directory_manager, memory_manager, gpu_manager=None, hdd_manager=None, status_display=None):
        self.dir_manager = directory_manager
        self.mem_manager = memory_manager
        self.gpu_manager = gpu_manager
        self.hdd_manager = hdd_manager
        self.status_display = status_display
        self.current_batch = 0
        self.processing_logger = logging.getLogger('processing_errors')
        self.system_logger = logging.getLogger('system_errors')
        self.current_utc = "2025-06-20 23:31:03"  # Updated timestamp
        self.user = "arullr001"
        self.total_combinations = 0
        self.processed_combinations = 0
        self.successful_combinations = 0

    def initialize_batch_tracking(self, total_combinations):
        """Initialize batch processing tracking"""
        print("\nDebug - Initializing batch tracking")
        self.total_combinations = total_combinations
        self.processed_combinations = 0
        self.successful_combinations = 0
        
        tracking_file = os.path.join(self.dir_manager.base_dir, 'batch_tracking.json')
        tracking_data = {
            'total_combinations': total_combinations,
            'processed_combinations': 0,
            'successful_combinations': 0,
            'start_time': self.current_utc,
            'last_update': self.current_utc,
            'status': 'initialized'
        }
        
        print(f"Debug - Created tracking file: {tracking_file}")
        with open(tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=4)

    def process_batch(self, df, param_combinations, batch_start, batch_size, max_workers=4):
        """Processes a batch of parameter combinations"""
        print(f"\nDebug - Starting batch processing:")
        print(f"Initial batch size: {batch_size}")
        
        # Calculate batch size
        available_memory = self.mem_manager.get_available_ram()
        required_memory_per_combo = df.memory_usage().sum() * 3
        adjusted_batch_size = min(
            batch_size,
            int(available_memory / (required_memory_per_combo * 1.5)),
            len(param_combinations) - batch_start
        )

        batch_end = min(batch_start + adjusted_batch_size, len(param_combinations))
        current_combinations = param_combinations[batch_start:batch_end]
        
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # Submit tasks
            for combo in current_combinations:
                futures.append(executor.submit(
                    process_param_combo,
                    (df.copy(), *combo)
                ))

            # Process results
            with tqdm(total=len(futures), desc=f"Batch {self.current_batch + 1} Progress") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result and result.get('trade_count', 0) > 0:
                            results.append(result)
                            self.successful_combinations += 1
                            
                            # Update status display if available
                            if self.status_display:
                                self.status_display.update(
                                    self.processed_combinations,
                                    top_combo=result
                                )
                        
                        self.processed_combinations += 1
                        pbar.update(1)
                        
                    except Exception as e:
                        error_msg = f"Error in batch {self.current_batch + 1}: {str(e)}"
                        print(f"\nDebug - Error processing task:")
                        print(error_msg)
                        self.processing_logger.error(error_msg)
                        pbar.update(1)

        # Save batch results to CSV explicitly - FIXED: Added explicit call to save_batch_results
        if results:
            print(f"\nDebug - Saving {len(results)} results from batch {self.current_batch}")
            self.save_batch_results(results, self.current_batch)
        else:
            print(f"\nDebug - No valid results in batch {self.current_batch} to save")

        # Save batch data to HDD manager if available
        if self.hdd_manager and results:
            batch_data = {
                'results': results,
                'timestamp': self.current_utc,
                'batch_number': self.current_batch
            }
            self.hdd_manager.save_batch_to_disk(batch_data, self.current_batch)

        # Update progress tracking
        self._update_progress()

        # Cleanup
        if self.hdd_manager:
            self.hdd_manager.cleanup_temp_files()
        cleanup_memory()
        
        self.current_batch += 1
        return results

    def _process_gpu_batch(self, df, combinations, gpu_id):
        """Process a batch on specific GPU"""
        gpu_name = "NVIDIA" if gpu_id == 0 else "Intel"
        print(f"\nProcessing on {gpu_name} GPU")
        
        try:
            # Set GPU context
            if gpu_id == 0 and cuda.is_available():
                cuda.select_device(0)
            
            results = []
            for combo in combinations:
                try:
                    result = process_param_combo((df, *combo))
                    if result:
                        results.append(result)
                except Exception as e:
                    self.processing_logger.error(f"Error processing combination on {gpu_name}: {e}")
            
            return results
            
        except Exception as e:
            self.processing_logger.error(f"Error in {gpu_name} GPU batch: {e}")
            return []

    def _update_progress(self):
        """Update progress tracking file"""
        try:
            tracking_file = os.path.join(self.dir_manager.base_dir, 'batch_tracking.json')
            progress_data = {
                'total_combinations': self.total_combinations,
                'processed_combinations': self.processed_combinations,
                'successful_combinations': self.successful_combinations,
                'completion_percentage': (self.processed_combinations / self.total_combinations) * 100,
                'last_update': self.current_utc,
                'status': 'running'
            }
            
            with open(tracking_file, 'w') as f:
                json.dump(progress_data, f, indent=4)
                
        except Exception as e:
            print(f"Debug - Error updating progress: {str(e)}")

    def save_batch_results(self, results, batch_num):
        """Saves batch results to CSV with enhanced debugging"""
        try:
            print(f"\nDebug - Starting save_batch_results for batch {batch_num}")
            batch_file = os.path.join(self.dir_manager.csv_dumps_dir, f'batch_{batch_num}.csv')
            print(f"Debug - Target file path: {batch_file}")
            
            results_data = []
            for r in results:
                if r['trades']:  # Only include valid results
                    row = {
                        'atr_length': r['parameters']['atr_length'],
                        'factor': r['parameters']['factor'],
                        'buffer_multiplier': r['parameters']['buffer_multiplier'],
                        'hard_stop_distance': r['parameters']['hard_stop_distance'],
                        'total_profit': r['total_profit'],
                        'trade_count': r['trade_count'],
                        'win_rate': r['win_rate'],
                        'profit_factor': r.get('profit_factor', 0),
                        'sharpe_ratio': r.get('sharpe_ratio', 0), 
                        'max_drawdown': r.get('max_drawdown', 0),
                        'risk_adjusted_return': r.get('risk_adjusted_return', 0),
                        'avg_trade_duration': r.get('avg_trade_duration', 0)
                    }
                    results_data.append(row)

            if results_data:
                df = pd.DataFrame(results_data)
                df.to_csv(batch_file, index=False)
                print(f"Debug - Successfully saved {len(results_data)} results to {batch_file}")
                
                # Verify file was created
                if os.path.exists(batch_file):
                    file_size = os.path.getsize(batch_file)
                    print(f"Debug - Verified file exists with size: {file_size} bytes")
                else:
                    print(f"Debug - ERROR: File was not created: {batch_file}")
                
                # Save metadata
                metadata_file = os.path.join(
                    self.dir_manager.csv_dumps_dir, 
                    f'batch_{batch_num}_metadata.json'
                )
                metadata = {
                    'batch_number': batch_num,
                    'processed_at': self.current_utc,
                    'processed_by': self.user,
                    'combinations_processed': len(results),
                    'valid_results': len(results_data),
                    'memory_usage_mb': self.mem_manager.get_current_ram_usage() / (1024 * 1024)
                }
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=4)
                print(f"Debug - Saved metadata to {metadata_file}")
                
                return True
                
            print("Debug - No valid results to save")
            return False

        except Exception as e:
            error_msg = f"Error saving batch {batch_num}: {str(e)}\n{traceback.format_exc()}"
            print(f"\nDebug - Error saving batch results:")
            print(error_msg)
            self.processing_logger.error(error_msg)
            return False

    def merge_batch_results(self):
        """Merges all batch results into final results with enhanced debugging"""
        print("\nDebug - Starting batch results merge")
        all_results = []
        batch_files = glob.glob(os.path.join(self.dir_manager.csv_dumps_dir, 'batch_*.csv'))
        
        print(f"Debug - Found {len(batch_files)} batch files")
        if len(batch_files) == 0:
            print(f"Debug - Looking in directory: {self.dir_manager.csv_dumps_dir}")
            if os.path.exists(self.dir_manager.csv_dumps_dir):
                all_files = os.listdir(self.dir_manager.csv_dumps_dir)
                print(f"Debug - All files in directory: {all_files}")
            else:
                print(f"Debug - Directory does not exist: {self.dir_manager.csv_dumps_dir}")
        
        try:
            for file in batch_files:
                try:
                    print(f"Processing batch file: {file}")
                    df = pd.read_csv(file)
                    print(f"Debug - Loaded file with {len(df)} rows")
                    all_results.append(df)
                except Exception as e:
                    error_msg = f"Error reading batch file {file}: {str(e)}"
                    print(f"Debug - Error reading batch file:")
                    print(error_msg)
                    self.processing_logger.error(error_msg)

            if all_results:
                final_df = pd.concat(all_results, ignore_index=True)
                final_df.sort_values('total_profit', ascending=False, inplace=True)
                
                final_results_file = os.path.join(
                    self.dir_manager.final_results_dir, 
                    'all_results.csv'
                )
                final_df.to_csv(final_results_file, index=False)
                print(f"Debug - Saved merged results ({len(final_df)} rows) to {final_results_file}")
                
                final_metadata = {
                    'processed_at': self.current_utc,
                    'processed_by': self.user,
                    'total_combinations_processed': self.processed_combinations,
                    'successful_combinations': self.successful_combinations,
                    'total_valid_results': len(final_df),
                    'best_profit': float(final_df['total_profit'].max()),
                    'average_profit': float(final_df['total_profit'].mean()),
                    'completion_time': self.current_utc
                }
                
                metadata_file = os.path.join(
                    self.dir_manager.final_results_dir, 
                    'final_metadata.json'
                )
                with open(metadata_file, 'w') as f:
                    json.dump(final_metadata, f, indent=4)
                print(f"Debug - Saved final metadata to {metadata_file}")
                
                return final_df
            
            print("Debug - No results to merge")
            return pd.DataFrame()

        except Exception as e:
            error_msg = f"Error merging batch results: {str(e)}\n{traceback.format_exc()}"
            print(f"\nDebug - Error merging results:")
            print(error_msg)
            self.system_logger.error(error_msg)
            return pd.DataFrame()


CURRENT_UTC = "2025-06-12 22:24:39"
CURRENT_USER = "arullr001"

class ResultsManager:
    """Manages the generation and storage of results and analysis"""
    def __init__(self, directory_manager):
        self.dir_manager = directory_manager
        self.processing_logger = logging.getLogger('processing_errors')
        self.system_logger = logging.getLogger('system_errors')
        self.cached_results = {}
        self.current_utc = CURRENT_UTC
        self.user = CURRENT_USER



    def rank_parameter_combinations(self, final_results_df, filters):
        """Rank parameter combinations based on multiple metrics with optional filtering"""
        try:
            # Start with the full dataset
            filtered_df = final_results_df.copy()
        
            # Apply filters based on preferences
            if filters['use_drawdown'] and filters['max_drawdown'] is not None:
                filtered_df = filtered_df[
                    filtered_df['max_drawdown'] <= filters['max_drawdown']
                ]
            
            if filters['use_profit'] and filters['min_profit'] is not None:
                filtered_df = filtered_df[
                    filtered_df['total_profit'] >= filters['min_profit']
                ]

            if filtered_df.empty:
                print("\nNo parameter combinations meet the filtering criteria.")
                print("Consider adjusting the filtering thresholds or running without filters.")
                return pd.DataFrame()


            # Apply minimum trade count filter - FIXED: Use filters['min_trades'] instead of hardcoded value
            min_trades = filters.get('min_trades', 30) if filters.get('use_min_trades', True) else 30
            print(f"Combinations before trade filter: {len(filtered_df)}")
            print(f"Using minimum trade count filter: {min_trades}")
            filtered_df = filtered_df[filtered_df['trade_count'] >= min_trades]

            if filtered_df.empty:
                print(f"\nNo parameter combinations have {min_trades}+ trades.")
                print("Consider relaxing filter criteria or expanding parameter ranges.")
                return pd.DataFrame()
    
            print(f"Combinations after trade filter: {len(filtered_df)}")
        
        
            # Calculate composite score
            filtered_df['composite_score'] = (
                filtered_df['profit_factor'] * 0.4 +    # 40% weight
                filtered_df['sharpe_ratio'] * 0.3 +     # 30% weight
                filtered_df['win_rate'] * 0.3           # 30% weight
            )

            # Sort by profit factor first, then use composite score for tie-breaking
            ranked_df = filtered_df.sort_values(
                ['profit_factor', 'composite_score', 'win_rate'],
                ascending=[False, False, False]
            )

            # Get top 5 combinations
            top_5 = ranked_df.head(5)

            # Save detailed results for top 5
            detailed_results_dir = os.path.join(self.dir_manager.final_results_dir, 'top_5_combinations')
            os.makedirs(detailed_results_dir, exist_ok=True)

            # Save results with all metrics
            results_file = os.path.join(detailed_results_dir, 'top_5_ranked_results.csv')
            top_5.to_csv(results_file, index=False)

            # Create detailed summary
            summary_data = {
                'analysis_timestamp': self.current_utc,
                'analyzed_by': self.user,
                'filtering_criteria': {
                    'filters_used': bool(filters['use_drawdown'] or filters['use_profit']),
                    'drawdown_filter': {
                        'used': filters['use_drawdown'],
                        'threshold': filters['max_drawdown']
                    },
                    'profit_filter': {
                        'used': filters['use_profit'],
                        'threshold': filters['min_profit']
                    }
                },
                'total_combinations_analyzed': len(final_results_df),
                'combinations_after_filtering': len(filtered_df),
                'top_5_combinations': []
            }

            print("\nTop 5 Parameter Combinations:")
            for idx, row in top_5.iterrows():
                combo_data = {
                    'rank': idx + 1,
                    'parameters': {
                        'atr_length': int(row['atr_length']),
                        'factor': float(row['factor']),
                        'buffer_multiplier': float(row['buffer_multiplier']),
                        'hard_stop_distance': float(row['hard_stop_distance'])
                    },
                    'metrics': {
                        'profit_factor': float(row['profit_factor']),
                        'sharpe_ratio': float(row['sharpe_ratio']),
                        'win_rate': float(row['win_rate']),
                        'max_drawdown': float(row['max_drawdown']),
                        'total_profit': float(row['total_profit']),
                        'expectancy': float(row.get('expectancy', 0)),
                        'trade_count': int(row['trade_count']),
                        'avg_trade_duration': float(row['avg_trade_duration'])
                    }
                }
                summary_data['top_5_combinations'].append(combo_data)
            
                print(f"\nRank {idx + 1}:")
                print(f"Parameters: ATR={row['atr_length']}, Factor={row['factor']:.2f}, "
                      f"Buffer={row['buffer_multiplier']:.2f}, Stop={row['hard_stop_distance']}")
                print(f"Profit Factor: {row['profit_factor']:.2f}")
                print(f"Sharpe Ratio: {row['sharpe_ratio']:.2f}")
                print(f"Win Rate: {row['win_rate']:.2%}")
                print(f"Max Drawdown: {row['max_drawdown']:.2%}")
                print(f"Net Profit: {row['total_profit']:.2f}")
                print(f"Total Trades: {row['trade_count']}")
                print(f"Avg Duration: {row['avg_trade_duration']:.2f} hours")

            # Save summary to JSON
            summary_file = os.path.join(detailed_results_dir, 'top_5_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=4)

            return top_5

        except Exception as e:
            self.processing_logger.error(f"Error ranking parameter combinations: {str(e)}")
            print(f"\nError ranking parameter combinations: {str(e)}")
            return pd.DataFrame()
    



    def create_performance_summary(self, final_results_df):
        """Creates and saves a performance summary of the results"""
        try:
            if final_results_df.empty:
                print("\nNo results to summarize.")
                return

            summary = {
                'timestamp': self.current_utc,
                'generated_by': self.user,
                'total_combinations_tested': len(final_results_df),
                'best_performance': {
                    'total_profit': float(final_results_df['total_profit'].max()),
                    'parameters': final_results_df.iloc[0].to_dict(),
                },
                'average_performance': {
                    'profit': float(final_results_df['total_profit'].mean()),
                    'trade_count': float(final_results_df['trade_count'].mean()),
                    'win_rate': float(final_results_df['win_rate'].mean()),
                    'sharpe_ratio': float(final_results_df['sharpe_ratio'].mean()),
                    'profit_factor': float(final_results_df['profit_factor'].mean()),
                },
                'profit_distribution': {
                    'min': float(final_results_df['total_profit'].min()),
                    'max': float(final_results_df['total_profit'].max()),
                    'median': float(final_results_df['total_profit'].median()),
                    'std': float(final_results_df['total_profit'].std()),
                },
                'risk_metrics': {
                    'max_drawdown_min': float(final_results_df['max_drawdown'].min()),
                    'max_drawdown_avg': float(final_results_df['max_drawdown'].mean()),
                    'max_drawdown_max': float(final_results_df['max_drawdown'].max()),
                    'sharpe_ratio_best': float(final_results_df['sharpe_ratio'].max()),
                    'profit_factor_best': float(final_results_df['profit_factor'].max())
                }
            }

            # Save summary to JSON
            summary_file = os.path.join(self.dir_manager.final_results_dir, 'performance_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4)
            print(f"\nPerformance summary saved to: {summary_file}")

        except Exception as e:
            self.processing_logger.error(f"Error creating performance summary: {str(e)}")
            print(f"\nError creating performance summary: {str(e)}")

    def generate_summary_report(self, params, filters, final_results_df, top_combinations, start_time, file_paths):
        """
        Generates a comprehensive summary report for the backtest run
    
        Args:
            params: Dictionary containing parameter ranges
            filters: Dictionary containing filter settings
            final_results_df: DataFrame with all results
            top_combinations: DataFrame with top combinations
            start_time: Start time of the run
            file_paths: List of data files processed
        """
        try:
            end_time = time.time()
            processing_time = end_time - start_time
            results_file = os.path.join(self.dir_manager.final_results_dir, 'summary_report.txt')
        
            with open(results_file, 'w') as f:
                # Header
                f.write("="*70 + "\n")
                f.write(" SUPERTREND BACKTESTER: SUMMARY REPORT ".center(70) + "\n")
                f.write("="*70 + "\n\n")
            
                # Run Information
                f.write("RUN INFORMATION:\n")
                f.write("-"*70 + "\n")
                f.write(f"Run Date & Time (UTC): {self.current_utc}\n")
                f.write(f"Run By: {self.user}\n")
                f.write(f"Results Directory: {os.path.abspath(self.dir_manager.base_dir)}\n\n")
            
                # Data Files
                f.write("DATA FILES PROCESSED:\n")
                f.write("-"*70 + "\n")
                for i, file_path in enumerate(file_paths, 1):
                    f.write(f"{i}. {os.path.basename(file_path)}\n")
                f.write("\n")
            
                # Parameter Ranges
                f.write("PARAMETER RANGES TESTED:\n")
                f.write("-"*70 + "\n")
                step_atr = params['atr_lengths'][1]-params['atr_lengths'][0] if len(params['atr_lengths'])>1 else 0
                f.write(f"ATR Length: {min(params['atr_lengths'])} to {max(params['atr_lengths'])} (step {step_atr})\n")
        
                step_factor = params['factors'][1]-params['factors'][0] if len(params['factors'])>1 else 0
                f.write(f"Factor: {min(params['factors']):.2f} to {max(params['factors']):.2f} (step {step_factor:.2f})\n")
        
                step_buffer = params['buffers'][1]-params['buffers'][0] if len(params['buffers'])>1 else 0
                f.write(f"Buffer Multiplier: {min(params['buffers']):.2f} to {max(params['buffers']):.2f} (step {step_buffer:.2f})\n")
        
                step_stop = params['stops'][1]-params['stops'][0] if len(params['stops'])>1 else 0
                f.write(f"Hard Stop Distance: {min(params['stops'])} to {max(params['stops'])} (step {step_stop})\n")
        
                f.write(f"Total Combinations Tested: {params['total_combinations']:,}\n\n")
            
                # Filtering Settings
                f.write("FILTERING SETTINGS:\n")
                f.write("-"*70 + "\n")
            
                # Min Trade Count Filter
                min_trades = filters.get('min_trades', 30)
                f.write(f"Min Trade Count Filter: {'Enabled' if filters.get('use_min_trades', True) else 'Disabled'}")
                if filters.get('use_min_trades', True):
                    f.write(f" (threshold: {min_trades})\n")
                else:
                    f.write("\n")
            
                # Max Drawdown Filter
                f.write(f"Max Drawdown Filter: {'Enabled' if filters.get('use_drawdown', False) else 'Disabled'}")
                if filters.get('use_drawdown', False):
                    f.write(f" (threshold: {filters.get('max_drawdown', 0):.2%})\n")
                else:
                    f.write("\n")
            
                # Min Profit Filter
                f.write(f"Min Profit Filter: {'Enabled' if filters.get('use_profit', False) else 'Disabled'}")
                if filters.get('use_profit', False):
                    f.write(f" (threshold: {filters.get('min_profit', 0):.2%})\n")
                else:
                    f.write("\n")
            
                if not final_results_df.empty:
                    f.write(f"Combinations After All Filtering: {len(final_results_df):,}\n\n")
                else:
                    f.write("No combinations passed all filters\n\n")
            
                # Top Combinations
                f.write("TOP PARAMETER COMBINATIONS:\n")
                f.write("-"*70 + "\n")
                if not top_combinations.empty:
                    for i, (_, row) in enumerate(top_combinations.iterrows(), 1):
                        f.write(f"Rank {i}:\n")
                        f.write(f"  Parameters: ATR={int(row['atr_length'])}, Factor={float(row['factor']):.2f}, ")
                        f.write(f"Buffer={float(row['buffer_multiplier']):.2f}, Stop={int(row['hard_stop_distance'])}\n")
                        f.write(f"  Profit Factor: {float(row.get('profit_factor', 0)):.2f}\n")
                        f.write(f"  Win Rate: {float(row.get('win_rate', 0)):.2%}\n")
                        f.write(f"  Trade Count: {int(row.get('trade_count', 0))}\n")
                        f.write(f"  Net Profit: {float(row.get('total_profit', 0)):.2f}\n")
                        f.write(f"  Max Drawdown: {float(row.get('max_drawdown', 0)):.2%}\n")
                        f.write(f"  Avg Trade Duration: {float(row.get('avg_trade_duration', 0)):.2f} hours\n")
                        f.write(f"  Avg Candle Count: {float(row.get('avg_candle_count', 0)):.1f}\n")
                        f.write("\n")
                else:
                    f.write("No top combinations found\n\n")
            
                # Performance Summary
                f.write("PERFORMANCE SUMMARY:\n")
                f.write("-"*70 + "\n")
                if not final_results_df.empty:
                    f.write(f"Total Valid Results: {len(final_results_df):,}\n")
                    f.write(f"Best Profit: {final_results_df['total_profit'].max():.2f}\n")
                    f.write(f"Average Profit: {final_results_df['total_profit'].mean():.2f}\n")
                    f.write(f"Best Win Rate: {final_results_df['win_rate'].max():.2%}\n")
                    f.write(f"Best Profit Factor: {final_results_df['profit_factor'].max():.2f}\n")
                
                    if 'sharpe_ratio' in final_results_df.columns:
                        f.write(f"Best Sharpe Ratio: {final_results_df['sharpe_ratio'].max():.2f}\n")
                
                    f.write(f"Max Drawdown Range: {final_results_df['max_drawdown'].min():.2%} to ")
                    f.write(f"{final_results_df['max_drawdown'].max():.2%}\n\n")
                else:
                    f.write("No valid results to summarize\n\n")
            
                # System Information
                f.write("SYSTEM INFORMATION:\n")
                f.write("-"*70 + "\n")
                hours = int(processing_time // 3600)
                minutes = int((processing_time % 3600) // 60)
                seconds = int(processing_time % 60)
                f.write(f"Processing Time: {hours:02d}:{minutes:02d}:{seconds:02d}\n")
                f.write(f"GPU Utilization: {'Yes' if cuda.is_available() else 'No'}\n")
                f.write(f"Total Memory Used: {psutil.Process().memory_info().rss / (1024*1024):.2f} MB\n")
                f.write(f"Forced Exit After 2 Days: Enabled\n")  # Added new feature
                f.write("\n")
            
                # Time-Based Exit Information
                f.write("TIME-BASED EXIT SETTINGS:\n")
                f.write("-"*70 + "\n")
                f.write("Trades will be automatically closed after 2 days\n")
                f.write("This helps prevent excessively long positions and limits risk exposure\n")
                f.write("Exit timing: 2 days after entry\n\n")
            
                # Footer
                f.write("="*70 + "\n")
                f.write(" END OF SUMMARY REPORT ".center(70) + "\n")
                f.write("="*70 + "\n")
        
            print(f"\nSummary report generated: {results_file}")
            return results_file
        
        except Exception as e:
            error_msg = f"Error generating summary report: {str(e)}\n{traceback.format_exc()}"
            print(f"\nError generating summary report: {str(e)}")
            self.processing_logger.error(error_msg)
            return None



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
    def __init__(self):
        self.logger = logging.getLogger('system_errors')
        self.gpus = self._get_all_gpus()
        self.gpu_queues = {}
        self.initialize_gpu_queues()

    def _get_all_gpus(self):
        gpus = []
        if cuda.is_available():
            try:
                # Get NVIDIA GPU
                cuda.get_current_device()
                gpus.append({
                    'id': 0,
                    'name': 'NVIDIA GeForce GTX 1050 Ti',
                    'memory': 4096,  # 4GB
                    'compute_capability': 'CUDA'
                })
                
                # Get Intel GPU
                gpus.append({
                    'id': 1,
                    'name': 'Intel HD Graphics 630',
                    'memory': 4096,  # 4GB
                    'compute_capability': 'OpenCL'
                })
            except Exception as e:
                self.logger.error(f"Error detecting GPUs: {e}")
        return gpus

    def initialize_gpu_queues(self):
        for gpu in self.gpus:
            self.gpu_queues[gpu['id']] = {
                'queue': [],
                'memory_available': gpu['memory'],
                'active_tasks': 0
            }

    def assign_task_to_gpu(self, task_size):
        available_gpus = sorted(
            self.gpus,
            key=lambda x: self.gpu_queues[x['id']]['memory_available'],
            reverse=True
        )
        
        if available_gpus:
            gpu = available_gpus[0]
            if self.gpu_queues[gpu['id']]['memory_available'] >= task_size:
                self.gpu_queues[gpu['id']]['active_tasks'] += 1
                self.gpu_queues[gpu['id']]['memory_available'] -= task_size
                return gpu['id']
        return None

    def release_gpu_task(self, gpu_id, task_size):
        if gpu_id in self.gpu_queues:
            self.gpu_queues[gpu_id]['active_tasks'] -= 1
            self.gpu_queues[gpu_id]['memory_available'] += task_size



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
    status_display = None  # Initialize outside try block for cleanup in finally
    try:
        start_time = time.time()
        print("=" * 50)
        print(" SUPER TREND STRATEGY BACKTESTER (OPTIMIZED) ".center(50, "="))
        print("=" * 50)
        print(f"Started at (UTC): 2025-06-14 16:35:07")
        print(f"User: arullr001")

        # Initialize managers
        dir_manager = DirectoryManager()
        mem_manager = MemoryManager()
    
        # Initialize GPU manager with fallback
        try:
            gpu_manager = GPUManager()
            if not gpu_manager.gpus:
                print("No GPUs available, falling back to CPU processing")
                gpu_manager = None
        except Exception as e:
            print(f"Error initializing GPU manager: {e}")
            gpu_manager = None
    
        # Initialize HDD manager
        try:
            hdd_manager = HDDManager(dir_manager.base_dir)
        except Exception as e:
            print(f"Error initializing HDD manager: {e}")
            hdd_manager = None

        print(f"\nCreated working directory: {dir_manager.base_dir}")
        print("Directory structure:")
        print(f"├── csv_dumps/")
        print(f"├── error_logs/")
        print(f"└── final_results/")

        # --- NEW FILE SELECTION SECTION ---
        print("\nPlease select one or more OHLC data files (CSV/XLSX/XLS) using the file picker...")
        selected_files = select_files_gui()
        if not selected_files:
            print("No files selected for processing. Exiting.")
            sys.exit(1)
        print(f"\nProcessing {len(selected_files)} file(s)")
        # --- END FILE SELECTION SECTION ---

        # Step 3: Get parameter inputs for 5-minute timeframe
        print("\nConfiguring parameters for 5-minute timeframe:")
        print("Recommended ranges for 5-minute timeframe:")
        print("ATR Length: 8-24")
        print("Factor: 0.3-1.5")
        print("Buffer: 0.1-0.5")
        print("Hard Stop: 10-50")
        
        params = get_parameter_inputs()

        # Step 4: Get filtering preferences
        print("\nSetting up filtering preferences...")
        filters = get_filtering_preferences()

        # Debug output to verify filters
        print("\nDebug: Filters configuration:")
        print(f"Filters: {filters}")

        # Store filtering preferences in metadata
        metadata = {
            'timestamp': "2025-06-14 16:44:31",
            'user': "arullr001",
            'filtering_preferences': filters,
            'parameters': params
        }

        # Save metadata
        metadata_file = os.path.join(dir_manager.base_dir, 'run_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

        # Generate parameter combinations
        param_combinations = list(product(
            params['atr_lengths'],
            params['factors'],
            params['buffers'],
            params['stops']
        ))

        # Parameter validation and warnings
        if len(param_combinations) < 100:
            print("\nWarning: Very few parameter combinations. Consider expanding ranges.")
            proceed = input("Continue anyway? (y/n): ").lower().strip() == 'y'
            if not proceed:
                sys.exit(0)
        elif len(param_combinations) > 50000:
            print("\nWarning: Very large number of combinations. This may take a long time.")
            proceed = input("Continue anyway? (y/n): ").lower().strip() == 'y'
            if not proceed:
                sys.exit(0)

        print("\nInitializing status display...")
        print(f"Debug: Filters being passed to StatusDisplay: {filters}")

        # Initialize status display
        status_display = StatusDisplay(
            params=params,
            total_combinations=len(param_combinations),
            filename=selected_files[0],  # First selected file
            filters=filters
        )

        # Initialize batch processor with status display
        batch_processor = BatchProcessor(
            directory_manager=dir_manager,
            memory_manager=mem_manager,
            gpu_manager=gpu_manager,
            hdd_manager=hdd_manager,
            status_display=status_display
        )

        results_manager = ResultsManager(dir_manager)

        # Initialize batch tracking
        batch_processor.initialize_batch_tracking(len(param_combinations))

        # Step 5: Process each file
        for file_path in selected_files:
            print(f"\nProcessing file: {os.path.basename(file_path)}")
            try:
                # Load and verify data
                df = load_ohlc_data(file_path)
                
                # Verify 5-minute timeframe
                time_diff = df.index[1] - df.index[0]
                if pd.Timedelta(minutes=5) != time_diff:
                    print(f"\nWarning: Data timeframe appears to be {time_diff}, not 5 minutes")
                    proceed = input("Continue anyway? (y/n): ").lower().strip() == 'y'
                    if not proceed:
                        continue

                print(f"\nDebug - Loaded data verification:")
                print(f"Shape: {df.shape}")
                print(f"Date range: {df.index.min()} to {df.index.max()}")
                print(f"Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")

                # Process combinations in batches
                batch_size = mem_manager.calculate_optimal_batch_size(
                    df.memory_usage().sum(),
                    len(param_combinations)
                )

                all_results = []
                for batch_start in range(0, len(param_combinations), batch_size):
                    batch_results = batch_processor.process_batch(
                        df,
                        param_combinations,
                        batch_start,
                        batch_size
                    )
                    
                    if batch_results:
                        valid_results = [
                            r for r in batch_results 
                            if all(metric in r for metric in ['max_drawdown', 'sharpe_ratio', 'win_rate'])
                        ]
                        
                        if valid_results:
                            all_results.extend(valid_results)
                    
                    # Clean up after each batch
                    cleanup_memory()

                # Process and save final results
                if all_results:
                    final_results_df = batch_processor.merge_batch_results()
                    
                    if not final_results_df.empty:
                        # Create performance summary
                        results_manager.create_performance_summary(final_results_df)
                        
                        # Rank parameter combinations with new filtering system
                        top_combinations = results_manager.rank_parameter_combinations(
                            final_results_df,
                            filters=filters
                        )
                else:
                    save_empty_results_file(
                        file_path, 
                        dir_manager.final_results_dir,
                        params,
                        "2025-06-14 16:35:07",
                        "arullr001",
                        len(param_combinations)
                    )

            except Exception as e:
                error_msg = f"Error processing file {file_path}: {str(e)}\n{traceback.format_exc()}"
                print(f"\nError processing file:")
                print(error_msg)
                logging.getLogger('processing_errors').error(error_msg)
                continue
        
        #Generate TEXT Summary
        try:
            if 'results_manager' in locals() and 'params' in locals():
                summary_file = results_manager.generate_summary_report(
                    params=params,
                    filters=filters,
                    final_results_df=final_results_df if 'final_results_df' in locals() else pd.DataFrame(),
                    top_combinations=top_combinations if 'top_combinations' in locals() else pd.DataFrame(),
                    start_time=start_time,
                    file_paths=selected_files
                )
                print(f"Summary report saved to: {summary_file}")
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
        
        
        # Final cleanup and summary
        cleanup_memory()
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("\n" + "=" * 50)
        print(" PROCESSING COMPLETE ".center(50, "="))
        print("=" * 50)
        print(f"Results directory: {dir_manager.base_dir}")
        print(f"Total processing time: {timedelta(seconds=int(processing_time))}")
        print(f"Started at: 2025-06-14 16:35:07")
        print(f"Completed at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        error_msg = f"System error: {str(e)}\n{traceback.format_exc()}"
        print("\nA system error occurred:")
        print(error_msg)
        logging.getLogger('system_errors').error(error_msg)
        sys.exit(1)
    finally:
        if status_display:
            status_display.cleanup()



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

